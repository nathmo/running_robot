#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a video montage: 3x4 grid of failures, then success clips.
Using imageio to create a real MP4 file.
"""

from pathlib import Path
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

TRAINING_DIR = Path(__file__).parent
OUTPUT_DIR = TRAINING_DIR / "montage"
OUTPUT_DIR.mkdir(exist_ok=True)
MONTAGE_PATH = OUTPUT_DIR / "training_montage.mp4"

# Videos to include
FAILURES = [
    "milestones/videos/m6_seed0_FALLS.mp4",
    "milestones/videos/m3_seed4_FALLS.mp4",
    "milestones/videos/m3_seed7_FALLS.mp4",
    "milestones/videos/m4_seed0_FALLS.mp4",
    "milestones/scripted_m3_fail.mp4",
    "runs/ab_cpg_cold_s0/dash.mp4",
    "runs/ab_f_cold_s0/dash.mp4",
    "runs/ab_cpg_m2_s0/dash.mp4",
    "runs/ab_f_m2_s0/dash.mp4",
    "runs/ab_cpg_nr_m2_s0/dash.mp4",
    "runs/ab_cpg_wide_m2_s0/dash.mp4",
    "runs/ab_cpg_m3warm_s0/dash.mp4",
]

SUCCESSES = [
    ("milestones/scripted_m2_demo.mp4", "Scripted Demo"),
    ("milestones/walk_fwd_easy_s0.mp4", "Walk Easy"),
    ("milestones/walk_fwd3_s0.mp4", "Walk v3"),
    ("milestones/keyframe_shipped_m3_s0.mp4", "m3 Shipped"),
]

def load_video_frames(video_path, num_frames=60, target_size=(400, 400)):
    """Load video frames by streaming."""
    path = TRAINING_DIR / video_path
    if not path.exists():
        print("    [skip] not found")
        return None

    try:
        reader = imageio.get_reader(str(path), 'ffmpeg')
        frames = []
        frame_idx = 0

        for frame in reader:
            # Convert to PIL, crop to square, resize
            img = Image.fromarray(frame)
            w, h = img.size
            s = min(w, h)
            x = (w - s) // 2
            y = (h - s) // 2
            img = img.crop((x, y, x + s, y + s))
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            frames.append(np.array(img))

            frame_idx += 1
            if len(frames) >= num_frames:
                break

        reader.close()

        # Pad if needed
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))

        print(f"    [ok] {len(frames)} frames")
        return frames

    except Exception as e:
        print(f"    [error] {str(e)[:50]}")
        return None

def create_grid(frame_lists, grid_w=4, grid_h=3):
    """Composite frame_lists into a grid."""
    cell_size = 400
    output_w = cell_size * grid_w
    output_h = cell_size * grid_h

    # Pad with black if needed
    while len(frame_lists) < grid_w * grid_h:
        frame_lists.append([np.zeros((cell_size, cell_size, 3), dtype=np.uint8)] * 60)

    frame_lists = frame_lists[:grid_w * grid_h]
    num_frames = 60

    grid_frames = []
    for frame_idx in range(num_frames):
        canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        for i, frame_list in enumerate(frame_lists):
            row = i // grid_w
            col = i % grid_w
            x = col * cell_size
            y = row * cell_size
            frame = frame_list[min(frame_idx, len(frame_list) - 1)]
            canvas[y:y + cell_size, x:x + cell_size] = frame

        grid_frames.append(canvas)

    return grid_frames

def create_success_frames(successes, num_frames=60, output_size=(1600, 1200)):
    """Create sequence of success videos with labels, resized to output_size."""
    all_frames = []

    for video_path, label in successes:
        frames = load_video_frames(video_path, num_frames=num_frames, target_size=(960, 720))
        if frames:
            for frame in frames:
                img = Image.fromarray(frame)
                # Add black bar for label
                canvas = Image.new('RGB', (960, 780), color=(0, 0, 0))
                canvas.paste(img, (0, 0))
                draw = ImageDraw.Draw(canvas)
                try:
                    font = ImageFont.truetype("arial.ttf", 28)
                except:
                    font = ImageFont.load_default()
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_x = (960 - text_w) // 2
                draw.text((text_x, 740), label, font=font, fill=(255, 255, 255))

                # Resize to match grid output size
                canvas = canvas.resize(output_size, Image.Resampling.LANCZOS)
                all_frames.append(np.array(canvas))

    return all_frames

def main():
    print("\n" + "="*70)
    print("  TRAINING MONTAGE VIDEO GENERATOR")
    print("="*70)

    # Load failure videos
    print(f"\n[1/3] Loading failure videos into 3x4 grid...")
    failure_frames_list = []
    for i, video_path in enumerate(FAILURES):
        print(f"  {i+1:2d}. {Path(video_path).name:40s}", end=" ", flush=True)
        frames = load_video_frames(video_path, num_frames=60, target_size=(400, 400))
        if frames:
            failure_frames_list.append(frames)

    print(f"\n  Creating grid from {len(failure_frames_list)} clips...")
    failure_grid_frames = create_grid(failure_frames_list, grid_w=4, grid_h=3)

    # Load success videos
    print(f"\n[2/3] Loading success videos...")
    for i, (video_path, label) in enumerate(SUCCESSES):
        print(f"  {i+1}. {label:40s}", end=" ", flush=True)
        _ = load_video_frames(video_path, num_frames=60, target_size=(960, 720))

    output_size = (failure_grid_frames[0].shape[1], failure_grid_frames[0].shape[0])
    success_frames = create_success_frames(SUCCESSES, num_frames=60, output_size=output_size)

    # Combine
    print(f"\n[3/3] Assembling video...")
    all_frames = failure_grid_frames + success_frames
    print(f"  Total frames: {len(all_frames)}")
    print(f"  Grid size: {failure_grid_frames[0].shape if failure_grid_frames else 'N/A'}")
    print(f"  Writing to {MONTAGE_PATH.name}...")

    writer = imageio.get_writer(str(MONTAGE_PATH), fps=24, codec='libx264', pixelformat='yuv420p')
    for i, frame in enumerate(all_frames):
        writer.append_data(frame)
        if (i + 1) % 60 == 0:
            progress = int(100 * (i + 1) / len(all_frames))
            print(f"    {progress}% ({i+1}/{len(all_frames)} frames)", flush=True)
    writer.close()

    duration_s = len(all_frames) / 24
    print(f"\n{'='*70}")
    print(f"SUCCESS!")
    print(f"  File: {MONTAGE_PATH}")
    print(f"  Duration: {duration_s:.1f}s")
    print(f"  Resolution: {all_frames[0].shape[1]}x{all_frames[0].shape[0]}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
