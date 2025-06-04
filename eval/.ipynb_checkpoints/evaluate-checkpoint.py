from __future__ import annotations
from pathlib import Path
import logging
import io

import hydra
import torch
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from clearml import Task

from datautils.datasets import SignLanguageDataset, EOS_TOKEN, SOS_TOKEN
from datautils.collate_functions import make_collate_fn
from eval.metrics import char_error_rate
from utils.logging_utils import setup_logging


def greedy_decode(model, frames, sos_idx, eos_idx, pad_idx, max_len: int = 256):
    frames_mask = frames.flatten(2).abs().sum(-1).eq(0)  # [1,T]
    ys = torch.tensor([[sos_idx]], device=frames.device)
    use_amp = frames.device.type == "cuda"
    with torch.no_grad():
        while ys.size(1) < max_len:
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(
                    frames, ys, memory_key_padding_mask=frames_mask
                )  # [1,L,V]
                next_tok = logits[:, -1].argmax(-1, True)  # [1,1]
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() in {eos_idx, pad_idx}:
                break
    return ys.squeeze(0)


@hydra.main(
    config_path="../experiments/cnn_transformer_resnet18/configs",
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    task = Task.init(
        project_name=cfg.clearml.project_name,
        task_name="Evaluation of " + cfg.clearml.task_name,
        reuse_last_task_id=False,
    )
    logger = task.get_logger()
    log = setup_logging(level="INFO")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Running evaluation on {device}")
    use_amp = device == "cuda"

    ds = SignLanguageDataset(
        cfg.dataset.test_data_dir,
        cfg.dataset.test_annotation_file,
        cfg.dataset.transform,
    )
    log.info(f"Dataset size: {len(ds)} samples")
    pad_idx = ds.vocab.pad_index
    dl = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=make_collate_fn(pad_idx)
    )

    cfg.model.vocab_size = len(ds.vocab)
    cfg.model.pad_idx = pad_idx
    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls(cfg.model)
    ckpt = Path(cfg.paths.ckpt_dir + "/model_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    log.info(f"Loaded checkpoint: {ckpt}")

    sos_idx = ds.vocab.stoi[SOS_TOKEN]
    eos_idx = ds.vocab.stoi[EOS_TOKEN]

    refs, hyps = [], []
    prog = tqdm(dl, desc="Decoding", ncols=100)
    for i, (frames, targets) in enumerate(prog, 1):
        frames = frames.to(device)
        with torch.autocast("cuda", enabled=use_amp):
            pred = greedy_decode(model, frames, sos_idx, eos_idx, pad_idx)

        ref_txt = "".join(ds.vocab.itos[t.item()] for t in targets[0][1:-1])
        hyp_txt = "".join(ds.vocab.itos[t] for t in pred[1:-1])
        refs.append(ref_txt)
        hyps.append(hyp_txt)

        if i <= 3:
            img_tensor = frames[0, 0].cpu()  # [C,H,W]
            img = img_tensor.permute(1, 2, 0).numpy()  # → [H,W,C]

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = img.clip(0, 1)

            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Ref: {ref_txt} | Hyp: {hyp_txt}")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            pil_img = Image.open(buf).convert("RGB")
            logger.report_image("Examples", f"sample_{i}", iteration=0, image=pil_img)

    cer = sum(char_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    logger.report_scalar("eval/cer", "final", cer, 0)
    log.info(f"Final CER = {cer:.3f}")


if __name__ == "__main__":
    main()
