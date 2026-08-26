#!/usr/bin/env python3
"""
Create a training montage video showing failure diversity + success.
3x4 grid of failures (10s) → transition → successful runs.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Tuple
import re

TRAINING_DIR = Path(__file__).parent
RUNS_DIR = TRAINING_DIR / "runs"
MILESTONES_DIR = TRAINING_DIR / "milestones"
OUTPUT_DIR = TRAINING_DIR / "montage"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except:
        return 0

def find_failure_videos() -> List[Path]:
    """Find videos that look like failures."""
    failures = []

    # Explicit FALLS videos
    for video in MILESTONES_DIR.glob("videos/*FALLS.mp4"):
        failures.append(video)

    # Add specific script failures
    script_fails = [
        MILESTONES_DIR / "scripted_m3_fail.mp4",
    ]
    for f in script_fails:
        if f.exists():
            failures.append(f)

    # Sample diverse runs - take first few from each experiment type
    # These are all early/exploratory, so they likely contain failures
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if run_dir.is_dir() and ("cold" in run_dir.name or "ab_" in run_dir.name or "ankle2" in run_dir.name):
            video = run_dir / "dash.mp4"
            if video.exists():
                failures.append(video)
                if len(failures) >= 12:
                    break

    # Fallback: if we still don't have enough, add any run videos
    if len(failures) < 12:
        for run_dir in sorted(RUNS_DIR.iterdir()):
            video = run_dir / "dash.mp4"
            if video.exists():
                failures.append(video)
                if len(failures) >= 12:
                    break

    return list(set(failures))[:12]  # Return 12 unique failures

def find_success_videos() -> List[Path]:
    """Find videos that show success."""
    successes = []

    # Good milestones
    good_milestones = [
        "scripted_m2_demo.mp4",
        "scripted_m2.mp4",
        "walk_fwd_easy_s0.mp4",
        "walk_fwd3_s0.mp4",
        "walk_fwd_s0.mp4",
        "ladder_m2_s0.mp4",
        "teleop_v5_warm.mp4",
        "keyframe_shipped_m3_s0.mp4",
        "keyframe_balanced_m3_s0.mp4",
    ]

    for name in good_milestones:
        p = MILESTONES_DIR / name
        if p.exists():
            successes.append(p)

    # Sample successful runs (m2+, seed 0)
    for run_dir in RUNS_DIR.iterdir():
        if run_dir.is_dir() and ("m2" in run_dir.name or "m3" in run_dir.name):
            video = run_dir / "dash.mp4"
            if video.exists() and "cold" not in run_dir.name:
                successes.append(video)
                if len(successes) >= 12:
                    break

    return successes[:8]  # Return 8 success videos

def crop_and_scale(input_path: str, output_path: str, duration: float, scale: str = "640x480"):
    """Crop video to square and scale it."""
    # Crop to square from center, then scale
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-vf", f"crop=min(iw\\,ih):min(iw\\,ih),scale={scale}",
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "28",
        "-y", output_path
    ], check=True, capture_output=True)

def create_grid_montage(videos: List[Path], output_path: str, grid_w: int, grid_h: int):
    """Create NxN grid montage from videos."""
    if len(videos) != grid_w * grid_h:
        raise ValueError(f"Expected {grid_w * grid_h} videos, got {len(videos)}")

    # Prepare: scale all videos to same size (3 seconds each for 10s montage)
    scaled_dir = OUTPUT_DIR / "scaled"
    scaled_dir.mkdir(exist_ok=True)

    scaled_videos = []
    for i, video in enumerate(videos):
        output = scaled_dir / f"scaled_{i:02d}.mp4"
        print(f"  Scaling {video.name}...")
        crop_and_scale(str(video), str(output), duration=3.0, scale="480x480")
        scaled_videos.append(output)

    # Build ffmpeg filter for grid
    # xstack example: [0][1]hstack=inputs=2[h1]; [2][3]hstack=inputs=2[h2]; [h1][h2]vstack=inputs=2[out]
    filter_parts = []

    # Horizontal stacks for each row
    for row in range(grid_h):
        start = row * grid_w
        end = start + grid_w
        row_videos = scaled_videos[start:end]
        row_label = f"row{row}"

        input_labels = "".join([f"[{i}]" for i in range(start, end)])
        filter_parts.append(f"{input_labels}hstack=inputs={grid_w}[{row_label}]")

    # Vertical stack of rows
    row_labels = "".join([f"[row{i}]" for i in range(grid_h)])
    filter_parts.append(f"{row_labels}vstack=inputs={grid_h}[out]")

    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"]
    for video in scaled_videos:
        cmd.extend(["-i", str(video)])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(output_path)
    ])

    print(f"Creating {grid_w}x{grid_h} montage...")
    subprocess.run(cmd, check=True)

def create_transition_montage(failures: List[Path], successes: List[Path], output_path: str):
    """Create full montage: 3x4 failures → successes."""

    print("\n=== TRAINING MONTAGE CREATOR ===")
    print(f"Building: 3x4 failure grid + success transition")

    # Step 1: Create failure grid (10s total, 3s per clip)
    print("\n[1/3] Creating failure grid montage...")
    failure_grid = OUTPUT_DIR / "grid_failures.mp4"
    create_grid_montage(failures, str(failure_grid), 4, 3)

    # Step 2: Concatenate success videos with crossfade transitions
    print("\n[2/3] Preparing success reels...")
    success_scaled = OUTPUT_DIR / "scaled_success"
    success_scaled.mkdir(exist_ok=True)

    success_videos_scaled = []
    for i, video in enumerate(successes):
        output = success_scaled / f"success_{i:02d}.mp4"
        print(f"  Scaling success {i+1}/{len(successes)}...")
        crop_and_scale(str(video), str(output), duration=2.0, scale="960x720")
        success_videos_scaled.append(output)

    # Create concat file
    concat_file = OUTPUT_DIR / "success_concat.txt"
    with open(concat_file, "w") as f:
        for video in success_videos_scaled:
            f.write(f"file '{video}'\n")

    success_reel = OUTPUT_DIR / "success_reel.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy",
        str(success_reel)
    ], check=True, capture_output=True)

    # Step 3: Concatenate failure grid + success with transition
    print("\n[3/3] Assembling final montage with transition...")

    concat_file = OUTPUT_DIR / "final_concat.txt"
    with open(concat_file, "w") as f:
        f.write(f"file '{failure_grid}'\n")
        f.write(f"file '{success_reel}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy",
        str(output_path)
    ], check=True, capture_output=True)

    print(f"\n✓ Montage created: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Scanning for training videos...")

    failures = find_failure_videos()
    successes = find_success_videos()

    print(f"\nFound {len(failures)} failure videos")
    print(f"Found {len(successes)} success videos")

    if len(failures) < 12:
        print(f"WARNING: Only found {len(failures)} failures, need 12 for grid")
    if len(successes) < 4:
        print(f"WARNING: Only found {len(successes)} successes, need at least 4")

    print("\nFailure videos:")
    for v in failures:
        print(f"  - {v.relative_to(TRAINING_DIR)}")

    print("\nSuccess videos:")
    for v in successes[:4]:
        print(f"  - {v.relative_to(TRAINING_DIR)}")

    output = str(OUTPUT_DIR / "training_montage.mp4")
    create_transition_montage(failures, successes, output)
