import argparse, tarfile, zipfile, tempfile, shutil
from pathlib import Path
from urllib.request import urlretrieve

DATASETS = {
    "slovo": {
        "url": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip",
        "type": "zip",
    },
}


def _extract(archive: Path, dst: Path, archive_type: str):
    print(f"Extracting -> {dst.relative_to(Path.cwd())}/")
    dst.mkdir(parents=True, exist_ok=True)
    if archive_type in {"tar.gz", "tar"}:
        mode = "r:gz" if archive_type == "tar.gz" else "r:"
        with tarfile.open(archive, mode) as tar:
            tar.extractall(dst)
    elif archive_type == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dst)
    else:
        raise ValueError(f"Unsupported archive type: {archive_type}")


# def _download(url: str, out: Path):
#     out.parent.mkdir(parents=True, exist_ok=True)
#     print(f"Downloading\n  {url}\n→ {out.relative_to(Path.cwd())}")
#     urlretrieve(url, out)
#     print("Download finished.")
def _download(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        dst_str = out.relative_to(Path.cwd())
    except ValueError:
        dst_str = out

    print(f"Downloading\n  {url}\n→ {dst_str}")
    urlretrieve(url, out)
    print("Download finished.")


def download_dataset(name: str, root: Path):
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'.  Run with --list to see options.")
    info = DATASETS[name]
    out_dir = root / name
    if out_dir.exists():
        print(f"• '{name}' already present at {out_dir}.  Skipping.")
        return

    tmp = Path(tempfile.mkdtemp())
    try:
        archive_path = tmp / "dataset_tmp"
        _download(info["url"], archive_path)
        _extract(archive_path, out_dir, info["type"])
        if (
            not (out_dir / "videos").exists()
            or not (out_dir / "annotations.tsv").exists()
        ):
            raise RuntimeError(
                "Archive does not match expected layout (videos/ + annotations.tsv)."
            )
        print(f"✅  '{name}' ready → {out_dir}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Sign-language dataset downloader")
    ap.add_argument("--name", help="Dataset key to download (see --list).")
    ap.add_argument(
        "--dest", default="data", help="Root folder to store datasets (default: ./data)"
    )
    ap.add_argument(
        "--list", action="store_true", help="Show available datasets & quit"
    )
    args = ap.parse_args()

    if args.list or not args.name:
        print("Available datasets:")
        for k in DATASETS:
            print(f"  {k:12} {DATASETS[k]['url']}")
        return

    download_dataset(args.name, Path(args.dest))


if __name__ == "__main__":
    main()
