import cv2, argparse
from pathlib import Path


def extract_video(video_path: Path, out_dir: Path, fps: int = 25):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(int(round(orig_fps / fps)), 1)
    idx = 0
    frame_id = 1
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            cv2.imwrite(str(out_dir / f"frame_{frame_id:04d}.jpg"), frame)
            frame_id += 1
        idx += 1
        ok, frame = cap.read()
    cap.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir", help="dir with *.mp4 videos")
    ap.add_argument("out_root", help="output root for frame folders")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    out_root = Path(args.out_root)
    for vid in video_dir.glob("*.mp4"):
        out_dir = out_root / vid.stem
        extract_video(vid, out_dir, args.fps)
