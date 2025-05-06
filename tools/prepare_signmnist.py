from pathlib import Path
import argparse, csv, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

LABEL2CHAR = {
    **{i: chr(ord("A") + i) for i in range(9)},  # A-I
    **{i: chr(ord("K") + i - 9) for i in range(9, 25)},
}  # K-Y


def row_to_img(row: pd.Series) -> Image.Image:
    """1 row → PIL.Image 28×28 (L mode)."""
    arr = row.values.astype(np.uint8).reshape(28, 28)
    return Image.fromarray(arr, mode="L")


def convert(csv_path: Path, out_root: Path):
    df = pd.read_csv(csv_path)
    annos = []
    videos = out_root / "videos"
    for idx, r in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        lbl = r["label"]
        pixels = r.drop("label")
        letter = LABEL2CHAR.get(lbl, "?")
        sample_id = f"{idx:06d}"
        clip_dir = videos / sample_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        img = row_to_img(pixels)
        img.save(clip_dir / "frame_0001.jpg")
        annos.append(f"{sample_id}\t{letter}\n")
    (out_root / "annotations.tsv").write_text("".join(annos), encoding="utf-8")
    print(f"✅ Done | samples: {len(df)} | root: {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train.csv")
    ap.add_argument("--out", default="data/signmnist")
    args = ap.parse_args()
    convert(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
