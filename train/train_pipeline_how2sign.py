from __future__ import annotations

from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datautils.dataset_how2sign import How2SignI3DDataset
from datautils.collate_how2sign import make_collate_fn

from utils import logging_utils, seed_utils
from utils.clearml_utils import get_logger, init_clearml
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import sacrebleu
from eval.metrics import char_error_rate, word_error_rate

def build_optimizer(model: nn.Module, lr: float):
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)


@torch.no_grad()
def step_validation(model, loader, loss_fn, device: str) -> float:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    use_amp = device == "cuda"
    for frames, targets in loader:
        frames, targets = frames.to(device), targets.to(device)
        frames_mask = frames.flatten(2).abs().sum(-1).eq(0)
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
def evaluate_batch_metrics(
    model,
    loader,
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
    vocab_or_sp,
    device: str,
):
    model.eval()
    total, correct = 0, 0
    cer_sum, wer_sum = 0.0, 0.0
    refs, hyps = [], []
    
    special_tokens = {pad_idx, eos_idx}
    
    def decode(ids: list[int]) -> str:
        ids = [i for i in ids if i not in special_tokens]
        if hasattr(vocab_or_sp, "decode_ids"):
            return vocab_or_sp.decode_ids(ids)
        else:
            return "".join(vocab_or_sp.itos[i] for i in ids)
    
    autocast_ctx = torch.amp.autocast("cuda") if device.startswith('cuda') else torch.nullcontext()
    
    with autocast_ctx:
        for frames, targets in tqdm(loader, desc="EvalMetrics", ncols=80):
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            preds = greedy_decode_batch(
                model=model,
                frames=frames,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                max_len=targets.size(1) * 2,
                device=device,
            )
            
            for ys, tgt in zip(preds, targets):
                hyp = decode(ys[1:].tolist())
                ref = decode(tgt[1:-1].tolist())
                refs.append(ref)
                hyps.append(hyp)
                correct += int(hyp == ref)
                cer_sum += char_error_rate(ref, hyp)
                wer_sum += word_error_rate(ref, hyp)
                total += 1
    
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    model.train()
    return {
        "exact_match": correct / total if total else 0.0,
        "cer": cer_sum / total if total else 0.0,
        "wer": wer_sum / total if total else 0.0,
        "bleu": bleu,
    }


@torch.no_grad()
def greedy_decode_batch(
    model,
    frames: torch.Tensor,        # [B, T_src, …]
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
    max_len: int,
    device: str,
):
    B = frames.size(0)
    src_mask = frames.flatten(2).abs().sum(-1).eq(0).to(device)  # [B, T_src]

    ys = torch.full((B, 1), sos_idx, device=device, dtype=torch.long)

    for _ in range(max_len - 1):
        logits = model(
            frames.to(device),
            ys,
            memory_key_padding_mask=src_mask,
        )  # -> [B, L, V]
        next_tok = logits[:, -1].argmax(-1, keepdim=True)  # [B,1]
        ys = torch.cat([ys, next_tok], dim=1)               # [B, L+1]

        if ((next_tok == eos_idx) | (next_tok == pad_idx)).all():
            break

    return ys  # [B, <=max_len]



@torch.no_grad()
def greedy_accuracy(
    model,
    loader,
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
    vocab_or_sp,
    device: str,
):

    model.eval()
    correct, total = 0, 0

    def decode(ids: list[int]) -> str:
        ids = [i for i in ids if i not in (pad_idx, eos_idx)]
        if hasattr(vocab_or_sp, "decode_ids"):
            return vocab_or_sp.decode_ids(ids)
        else:
            return "".join(vocab_or_sp.itos[i] for i in ids)

    for frames, targets in tqdm(loader, desc="GreedyEval", ncols=80):
        preds = greedy_decode_batch(
            model, frames, sos_idx, eos_idx, pad_idx,
            max_len=targets.size(1) * 2,
            device=device,
        )  # [B, L_pred]

        for ys, tgt in zip(preds, targets):
            hyp = decode(ys[1:].tolist())
            ref = decode(tgt[1:-1].tolist())
            correct += int(hyp == ref)
            total += 1

    model.train()
    return correct / total if total else 0.0

def fast_eval(model, loader, vocab_or_sp, sos_idx, eos_idx, pad_idx, device):
    model.eval()
    correct, total = 0, 0
    
    special_tokens = {pad_idx, eos_idx}
    
    def decode(ids: list[int]) -> str:
        ids = [i for i in ids if i not in special_tokens]
        if hasattr(vocab_or_sp, "decode_ids"):
            return vocab_or_sp.decode_ids(ids)
        else:
            return "".join(vocab_or_sp.itos[i] for i in ids)
    
    with torch.no_grad():
        autocast_ctx = torch.cuda.amp.autocast() if device.type == 'cuda' else torch.nullcontext()
        
        with autocast_ctx:
            for frames, targets in tqdm(loader, desc="FastEval", ncols=80):
                frames = frames.to(device, non_blocking=True)
                targts = targets.to(device, non_blocking=True)
                
                preds = greedy_decode_batch(
                    model, frames, sos_idx, eos_idx, pad_idx,
                    max_len=targets.size(1) * 2,
                    device=device,
                )
                
                for ys, tgt in zip(preds, targets):
                    hyp = decode(ys[1:].tolist())
                    ref = decode(tgt[1:-1].tolist())
                    correct += int(hyp == ref)
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


@hydra.main(config_path="../experiments/slt_transformer/configs", config_name="config.yaml", version_base=None)
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

    train_ds = How2SignI3DDataset(
        cfg.dataset.train_feature_dir,
        cfg.dataset.train_annotation_file,
        cfg.dataset.sp_model_path
    )
    val_ds = How2SignI3DDataset(
        cfg.dataset.val_feature_dir,
        cfg.dataset.val_annotation_file,
        cfg.dataset.sp_model_path
    )

    pad_idx = train_ds.pad_index
    collate = make_collate_fn(pad_idx)
    train_loader = DataLoader(train_ds, cfg.training.batch_size, True, collate_fn=collate, num_workers=8)
    val_loader = DataLoader(val_ds, cfg.training.batch_size, False, collate_fn=collate, num_workers=8)

    cfg.model.vocab_size = train_ds.vocab_size
    cfg.model.pad_idx = pad_idx
    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls(cfg.model).to(device)
    logger.info(model)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = build_optimizer(model, cfg.training.learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, cfg.training.patience // 2))
    sched_cfg = cfg.training.lr_scheduler

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.mode,
        factor=sched_cfg.factor,
        patience=sched_cfg.patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    ckpt_dir = Path(cfg.paths.ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "model_best.pt"
    latest_path = ckpt_dir / "model_latest.pt"

    sos_idx = train_ds.sp.bos_id()
    eos_idx = train_ds.sp.eos_id()

    best_val = float("inf")
    bad_epochs = 0
    global_step = 0
    best_cer   = float("inf")
    best_bleu  = -float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        logger.info(f"Epoch {epoch}/{cfg.training.epochs}")
        prog = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100)

        for frames, targets in prog:
            frames, targets = frames.to(device), targets.to(device)
            frames_mask = frames.flatten(2).abs().sum(-1).eq(0)
            inp, tgt = targets[:, :-1], targets[:, 1:]

            with torch.autocast("cuda", enabled=(device == "cuda")):
                logits = model(frames, inp, memory_key_padding_mask=frames_mask)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            prog.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % cfg.training.log_every_steps == 0:
                logger.info(f"step {global_step:>6}: train_loss={loss.item():.4f}")
                if clr_logger:
                    clr_logger.report_scalar("loss/train", "step", loss.item(), global_step)
            global_step += 1

        val_loss = step_validation(model, val_loader, loss_fn, device)
        #val_acc = greedy_accuracy(model, val_loader, sos_idx, eos_idx, pad_idx, train_ds.sp, device)
        metrics = evaluate_batch_metrics(
            model,
            val_loader,
            sos_idx, eos_idx, pad_idx,
            train_ds.sp if hasattr(train_ds, "sp") else train_ds.vocab, device)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch} | val_loss={val_loss:.4f} "
            f"| EM={metrics['exact_match']:.3f} "
            f"| CER={metrics['cer']:.3f} "
            f"| WER={metrics['wer']:.3f} "
            f"| BLEU={metrics['bleu']:.1f}"
        )
        #logger.info(f"Epoch {epoch} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
        #if clr_logger:
        #    clr_logger.report_scalar("loss/val_epoch", "epoch", val_loss, epoch)
        #    clr_logger.report_scalar("acc/val_epoch", "epoch", val_acc, epoch)
        if clr_logger:
            clr_logger.report_scalar("loss/val_epoch", "epoch", val_loss,epoch)
            clr_logger.report_scalar("acc/val_epoch","exact_match", metrics["exact_match"], epoch)
            clr_logger.report_scalar("cer/val_epoch","cer",metrics["cer"],epoch)
            clr_logger.report_scalar("wer/val_epoch","wer",metrics["wer"],epoch)
            clr_logger.report_scalar("bleu/val_epoch","bleu",metrics["bleu"],epoch)

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
            logger.info(f" ▸ No improvement — {bad_epochs}/{cfg.training.patience} bad epochs")

        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            cer_path = ckpt_dir / "model_best_cer.pt"
            save_checkpoint(model, cer_path)
            logger.info(f" ▸ New BEST_CER ckpt (CER={best_cer:.3f})")
            if task: upload_replace(task, "best_ckpt_cer", cer_path)

        if metrics["bleu"] > best_bleu:
            best_bleu = metrics["bleu"]
            bleu_path = ckpt_dir / "model_best_bleu.pt"
            save_checkpoint(model, bleu_path)
            logger.info(f" ▸ New BEST_BLEU ckpt (BLEU={best_bleu:.2f})")
            if task: upload_replace(task, "best_ckpt_bleu", bleu_path)

        if bad_epochs >= cfg.training.patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
