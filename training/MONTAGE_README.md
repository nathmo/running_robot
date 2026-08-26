# Training Montage: Failure Diversity Showcase

This generates an interactive gallery showing **12 failed experiments** vs **8 successful runs** from SpiderBot RL training.

## Quick Start

```bash
python serve_montage.py
```

This starts a local web server and opens `http://localhost:8000/training_montage.html` in your browser.

## What You'll See

### Failures Section (❌)
A 3×4 grid of different experimental approaches that didn't work:
- **Cold starts** (ab_cpg_cold, ab_f_cold): Raw policies with no curriculum
- **Architecture variants** (CPG vs Fourier): Different oscillator models
- **Hardware experiments** (rigid ankle, blade foot): Mechanical design sweeps
- **Ladder terrain** (m3/m4/m5): Progressive difficulty wall tests
- **Explicit failures**: Videos labeled `_FALLS` showing collapse modes

**The point:** ~68 total runs, but this grid shows **12 radically different hypotheses**. Each cell is a different question: *Does this control law work? Does this hardware? Does this terrain curriculum?*

### Success Section (✅)
Controllers that actually balanced and walked:
- **Scripted (m2)**: Open-loop kinematics proof + 3-gain pitch reflex
- **Walk_fwd lineage**: Warm-started RL policies at m2, m3, m3-easy
- **Keyframe variants**: Pitch-balanced stances (m3 shipped vs m3 balanced)
- **Teleop**: Sim2real joystick controller

**The gap:** Scripted hangs ~35s on the plant; successful RL policies reach 24.7+ seconds. The reflex alone was not enough—the robot needed learned reactive stepping.

## Design Notes

- Each video plays on hover; click to control playback
- Hovering shows the experiment name (e.g., "m3 Collapse", "CPG Cold")
- Descriptions explain what each section represents
- Mobile-responsive: adapts from 4 columns to 1 on small screens

## Customizing

Edit `training_montage.html` to change:
- **Video paths** (search `failureVideos` and `successVideos` arrays)
- **Grid layout** (change `grid-template-columns` in CSS)
- **Description text** (edit the `<p class="description">` blocks)

To add/remove videos, edit the arrays at the bottom of the file:

```javascript
const failureVideos = [
    { path: "../milestones/videos/...", name: "..." },
    // Add more here
];
```

The path is relative to where you run the server (i.e., the `training/` directory).

## Why This Format?

**Instead of a single MP4 file**, this interactive gallery:
- ✅ Lets you inspect any single experiment (pause, rewind, full-screen)
- ✅ Shows the **entire exploration process** (not just a highlight reel)
- ✅ Works with no ffmpeg dependency
- ✅ Mobile-friendly and shareable
- ✅ Fast to load (videos stream as you play them)

## File Structure

```
training/
├── serve_montage.py          # Run this to start the server
├── training_montage.html     # The gallery (open locally via server)
├── milestones/               # Key experiments (scripted, walk_fwd, etc.)
│   ├── *.mp4                 # Good runs
│   └── videos/               # Falls and parametric sweeps
└── runs/                      # All 68 experimental runs
    └── */dash.mp4            # Each run's final rollout
```

---

**The story in 10 seconds:** Dozens of approaches, most collapsed. A few found the right balance law + step reactivity. That's the training process.
