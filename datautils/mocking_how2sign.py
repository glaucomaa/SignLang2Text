import random
import string
from pathlib import Path

import numpy as np
import sentencepiece as spm

root = Path("mock_how2sign")
feat_dir = root / "train"
feat_dir.mkdir(parents=True, exist_ok=True)

chars = string.ascii_lowercase + " "
rows = ["id\ttranslation\n"]

for i in range(10):
    vid_id = f"sample_{i:02d}"
    feat = np.random.randn(random.randint(30, 60), 1024).astype("float32")
    np.save(feat_dir / f"{vid_id}.npy", feat)

    sent = "".join(random.choice(chars) for _ in range(random.randint(5, 15))).strip()
    rows.append(f"{vid_id}\t{sent}\n")

(root / "mock.tsv").write_text("".join(rows), encoding="utf-8")
print("fake features & TSV ready:", root)

_tmp = Path("mock_dummy.txt")
_tmp.write_text("dummy\n", encoding="utf-8")

spm.SentencePieceTrainer.train(
    input=str(_tmp),
    model_prefix="mock_sp",
    vocab_size=9,
    character_coverage=1.0,
    pad_id=3,
    pad_piece="<pad>",
)
(Path("mock_sp.model")).rename(root / "mock_sp.model")
_tmp.unlink()
print("mock vocab saved ->", root / "mock_sp.model")

