import argparse, os, glob, cv2, sys, math, torch, numpy as np
import torchvision.transforms as T
import torch.nn as nn
from tqdm import tqdm # cuz need
from pathlib import Path

def load_i3d_rgb(checkpoint, device="cuda", model_dir="pytorch-i3d"):
    sys.path.append(model_dir)          
    from pytorch_i3d import InceptionI3d
    net = InceptionI3d(400, in_channels=3)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.train(False).to(device) 
    return net    


def read_video(path):
    cap, frames = cv2.VideoCapture(path), []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames          


_resize_crop = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(), 
    T.Normalize([0.5]*3, [0.5]*3),
])
def frames_to_tensor(frames):
    fr_tensors = [_resize_crop(f) for f in frames]
    clip = torch.stack(fr_tensors)
    return clip.permute(1,0,2,3)


def video_to_tokens(frames, i3d, device, batch=8, clip_len=16, stride=8):
    pad_len = (8 - len(frames) % 8) % 8
    frames = frames + [frames[-1]] * pad_len

    windows: List[List[np.ndarray]] = []
    if len(frames) <= clip_len:
        windows.append(frames)
    else:
        for start in range(0, len(frames) - clip_len + 1, stride):
            windows.append(frames[start : start + clip_len])

    feats: List[torch.Tensor] = []
    with torch.no_grad():
        pb = tqdm(
            range(0, len(windows), batch),
            desc="      windows(batch)",
            leave=False,
            unit="batch",
        )
        for b in pb:
            clip_batch = [frames_to_tensor(w) for w in windows[b : b + batch]]
            clip_batch = torch.stack(clip_batch).to(device)
            fmap = i3d.extract_features_not_pool(clip_batch)
            pooled = fmap.mean((3, 4)).permute(0, 2, 1)  # [B,T/2,1024]
            feats.append(pooled.cpu())
    return torch.cat(feats, 0).reshape(-1, 1024) 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="videos", help="folder with *.mp4 etc.")
    ap.add_argument("--out_dir",    default="i3d_features", help="output npys")
    ap.add_argument("--model_dir",  default="pytorch-i3d",  help="clone of repo")
    ap.add_argument("--ckpt",       default="models/rgb_imagenet.pt",
                    help="RGB checkpoint inside repo/models/")
    ap.add_argument("--device",     default="cuda", choices=["cuda","cpu"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    i3d = load_i3d_rgb(args.ckpt, args.device, args.model_dir)
    #hot fix
    i3d.extract_features_not_pool = i3d.extract_features

    exts = ("*.mp4","*.mov","*.avi","*.mkv")
    videos = [f for ext in exts for f in glob.glob(os.path.join(args.videos_dir, ext))]
    if not videos:
        print("No video files found", args.videos_dir); return

    for vid in tqdm(sorted(videos), desc="Videos", unit="vid"):
        vid = Path(vid)
        stem = vid.stem
        out_path = Path(args.out_dir) / f"{stem}.npy"
        if out_path.exists():
            tqdm.write(f"[skip] {stem}")
            continue
        try:
            frames = read_video(vid)
            tokens = video_to_tokens(frames, i3d, args.device)
            np.save(out_path, tokens.numpy())
            tqdm.write(f"[OK]   {stem}: {tokens.shape}")
        except Exception as e:
            tqdm.write(f"[ERR]  {stem}: {e}")


if __name__ == "__main__":
    main()
