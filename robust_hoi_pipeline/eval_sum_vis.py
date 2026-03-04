import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _list_frames(frame_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    frames = [p for p in sorted(frame_dir.iterdir()) if p.suffix in exts]
    return frames


def _resolve_frame_dir(base_dir: Path, candidates):
    for rel in candidates:
        p = base_dir / rel
        if p.exists() and p.is_dir():
            return p
    return None


def _resize_to_height(img: np.ndarray, h: int):
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * (h / float(img.shape[0]))))
    w = max(2, w)
    if w % 2 == 1:
        w += 1
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _create_video(frame_dir: Path, out_video: Path, fps: int):
    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_video),
    ]
    print(f"Running command:\n{shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(args):
    foundation_dir = Path(args.foundation_dir)
    joint_opt_dir = Path(args.joint_opt_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    seq_name = out_dir.name

    foundation_frames_dir = _resolve_frame_dir(
        foundation_dir, ["nvdiffrast_overlay_frames", "overlay_frames"]
    )
    joint_opt_frames_dir = _resolve_frame_dir(
        joint_opt_dir, ["nvdiffrast_overlay_frames", "overlay_frames"]
    )
    gt_frames_dir = _resolve_frame_dir(
        gt_dir, ["gt_overlay_frames", "nvdiffrast_overlay_frames", "overlay_frames"]
    )

    if foundation_frames_dir is None:
        raise FileNotFoundError(f"No overlay frame dir found under {foundation_dir}")
    if joint_opt_frames_dir is None:
        raise FileNotFoundError(f"No overlay frame dir found under {joint_opt_dir}")
    if gt_frames_dir is None:
        raise FileNotFoundError(f"No overlay frame dir found under {gt_dir}")

    f_frames = _list_frames(foundation_frames_dir)
    j_frames = _list_frames(joint_opt_frames_dir)
    g_frames = _list_frames(gt_frames_dir)

    n = min(len(f_frames), len(j_frames), len(g_frames))
    if n == 0:
        raise RuntimeError(
            f"Empty input frames: foundation={len(f_frames)}, joint_opt={len(j_frames)}, gt={len(g_frames)}"
        )

    if args.rebuild and out_dir.exists():
        shutil.rmtree(out_dir)
    merge_frames_dir = out_dir / "eval_sum_vis_frames"
    merge_frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Merging {n} frames (foundation={len(f_frames)}, joint_opt={len(j_frames)}, gt={len(g_frames)})"
    )

    for i in tqdm(range(n), desc="Merging eval videos"):
        imgs = []
        for frame_path in (f_frames[i], j_frames[i], g_frames[i]):
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            imgs.append(img)

        target_h = min(im.shape[0] for im in imgs)
        imgs = [_resize_to_height(im, target_h) for im in imgs]

        sep = np.full((target_h, args.line_width, 3), args.line_gray, dtype=np.uint8)
        merged = np.concatenate([imgs[0], sep, imgs[1], sep, imgs[2]], axis=1)

        out_p = merge_frames_dir / f"{i:06d}.png"
        cv2.imwrite(str(out_p), merged)

    out_video = out_dir / f"../eval_sum_{seq_name}.mp4"
    _create_video(merge_frames_dir, out_video, args.fps)
    print(f"Saved merged video: {out_video}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation_dir", type=str, required=True, help="FoundationPose eval vis output directory")
    parser.add_argument("--joint_opt_dir", type=str, required=True, help="Joint-opt eval vis output directory")
    parser.add_argument("--gt_dir", type=str, required=True, help="GT eval vis output directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for merged visualization")
    parser.add_argument("--fps", type=int, default=6, help="Output video fps")
    parser.add_argument("--line_width", type=int, default=8, help="Separator line width in pixels")
    parser.add_argument("--line_gray", type=int, default=160, help="Separator gray value [0,255]")
    parser.add_argument("--rebuild", action="store_true", help="Clear output directory before writing")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
