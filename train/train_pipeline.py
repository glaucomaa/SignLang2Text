from __future__ import annotations

from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datautils.collate_functions import make_collate_fn
from datautils.datasets import EOS_TOKEN, SOS_TOKEN, SignLanguageDataset
from utils import logging_utils, seed_utils
from utils.clearml_utils import get_logger, init_clearml


def build_optimizer(model: nn.Module, lr: float):
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)


@torch.no_grad()
def step_validation(model, loader, loss_fn, device: str) -> float:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    use_amp = device == "cuda"
    for frames, targets in loader:
        frames, targets = frames.to(device), targets.to(device)
        frames_mask = frames.flatten(2).abs().sum(-1).eq(0)  # [B,T]

        inp, tgt = targets[:, :-1], targets[:, 1:]
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(frames, inp, memory_key_padding_mask=frames_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        ntok = tgt.ne(loss_fn.ignore_index).sum().item()
        tot_loss += loss.item() * ntok
        tot_tok += ntok
    model.train()
    return tot_loss / max(tot_tok, 1)


@torch.no_grad()
def greedy_accuracy(
    model,
    loader: DataLoader,
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
    vocab,
    device: str,
) -> float:
    model.eval()
    correct, total = 0, 0
    use_amp = device == "cuda"
    for frames, targets in loader:
        B = frames.size(0)
        frames = frames.to(device)
        frames_mask = frames.flatten(2).abs().sum(-1).eq(0)
        for i in range(B):
            clip = frames[i : i + 1]
            clip_mask = frames_mask[i : i + 1]
            target = targets[i]
            ys = torch.tensor([[sos_idx]], device=device)
            max_len = target.size(0) * 2
            while ys.size(1) < max_len:
                with torch.autocast("cuda", enabled=use_amp):
                    next_tok = model(clip, ys, memory_key_padding_mask=clip_mask)[
                        :, -1
                    ].argmax(-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() in (eos_idx, pad_idx):
                    break
            pred_txt = "".join(vocab.itos[t] for t in ys.squeeze(0)[1:-1])
            ref_txt = "".join(vocab.itos[t.item()] for t in target[1:-1])
            correct += int(pred_txt == ref_txt)
            total += 1
    model.train()
    return correct / total if total else 0.0


def save_checkpoint(model: nn.Module, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)


def upload_replace(task, name: str, file_path: Path):
    try:
        task.delete_artifact(name)
    except Exception:
        pass
    task.upload_artifact(name=name, artifact_object=str(file_path))


@hydra.main(
    config_path="../experiments/cnn_transformer_resnet18/configs",
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    seed_utils.set_seed(cfg.training.seed)
    run_dir = Path.cwd()
    logger = logging_utils.setup_logging(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    task = init_clearml(cfg)
    clr_logger = get_logger() if task else None
    if task is None:
        print("⚠️  ClearML disabled.")

    full_ds = SignLanguageDataset(
        cfg.dataset.data_dir, cfg.dataset.annotation_file, cfg.dataset.transform
    )
    val_size = int(len(full_ds) * float(getattr(cfg.dataset, "val_frac", 0.1)))
    train_ds, val_ds = random_split(
        full_ds,
        [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )

    pad_idx = full_ds.vocab.pad_index
    collate = make_collate_fn(pad_idx)
    train_loader = DataLoader(
        train_ds, cfg.training.batch_size, True, collate_fn=collate, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, cfg.training.batch_size, False, collate_fn=collate, num_workers=4
    )

    cfg.model.vocab_size = len(full_ds.vocab)
    cfg.model.pad_idx = pad_idx
    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls(cfg.model).to(device)
    logger.info(model)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = build_optimizer(model, cfg.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, cfg.training.patience // 2)
    )

    scaler = torch.GradScaler("cuda", enabled=device == "cuda")

    patience = int(getattr(cfg.training, "patience", 5))
    bad_epochs = 0
    best_val = float("inf")

    ckpt_dir = Path(cfg.paths.ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "model_best.pt"
    latest_path = ckpt_dir / "model_latest.pt"

    sos_idx = full_ds.vocab.stoi[SOS_TOKEN]
    eos_idx = full_ds.vocab.stoi[EOS_TOKEN]

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        logger.info(f"Epoch {epoch}/{cfg.training.epochs}")
        prog = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100)

        for frames, targets in prog:
            frames, targets = (
                frames.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            frames_mask = frames.flatten(2).abs().sum(-1).eq(0)  # [B,T]

            inp, tgt = targets[:, :-1], targets[:, 1:]

            with torch.autocast("cuda", enabled=device == "cuda"):
                logits = model(frames, inp, memory_key_padding_mask=frames_mask)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            prog.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % cfg.training.log_every_steps == 0:
                logger.info(f"step {global_step:>6}: train_loss={loss.item():.4f}")
                if clr_logger:
                    clr_logger.report_scalar(
                        "loss/train", "step", loss.item(), global_step
                    )
                    clr_logger.report_scalar(
                        "grad_norm", "step", gn.item(), global_step
                    )
            global_step += 1

        val_loss = step_validation(model, val_loader, loss_fn, device)
        val_acc = greedy_accuracy(
            model, val_loader, sos_idx, eos_idx, pad_idx, full_ds.vocab, device
        )
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
        if clr_logger:
            clr_logger.report_scalar("loss/val_epoch", "epoch", val_loss, epoch)
            clr_logger.report_scalar("acc/val_epoch", "epoch", val_acc, epoch)

        save_checkpoint(model, latest_path)
        if task:
            upload_replace(task, "latest_ckpt", latest_path)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, best_path)
            logger.info(f" ▸ New BEST ckpt saved (val_loss={val_loss:.4f})")
            if task:
                upload_replace(task, "best_ckpt", best_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            logger.info(f" ▸ No improvement — {bad_epochs}/{patience} bad epochs")

        if bad_epochs >= patience:
            logger.info(f"Early stopping triggered (patience={patience}).")
            break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
