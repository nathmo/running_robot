from __future__ import annotations

import argparse
import runpy
import time
from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "mujoco" / "running_robot_debug.xml"


def overlay_text(xml_path: Path) -> list[tuple[object, object, str, str]]:
    return [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            f"MJCF debug viewer: {xml_path.name}",
            "Edit the XML, save, and the viewer will reload it automatically.",
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "Enabled: joints, constraints, inertia, COM",
            "Press Esc or close the window to exit.",
        ),
    ]


def configure_viewer(handle: mujoco.viewer.Handle) -> None:
    handle.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
    handle.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
    handle.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    handle.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    handle.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
    handle.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = False
    handle.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


def load_model(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def run(xml_path: Path) -> None:
    last_good_mtime = None
    last_failed_mtime = None

    while True:
        try:
            model, data = load_model(xml_path)
        except Exception as exc:  # pragma: no cover - interactive tool
            print(f"Failed to load {xml_path}: {exc}")
            time.sleep(1.0)
            continue

        with mujoco.viewer.launch_passive(model, data) as handle:
            configure_viewer(handle)

            center = getattr(model.stat, "center", (0.0, 0.0, 0.0))
            extent = float(getattr(model.stat, "extent", 1.0) or 1.0)
            handle.cam.lookat[:] = center
            handle.cam.distance = max(1.0, extent * 2.5)
            handle.cam.azimuth = 135.0
            handle.cam.elevation = -25.0
            handle.set_texts(overlay_text(xml_path))

            last_good_mtime = xml_path.stat().st_mtime

            while handle.is_running():
                try:
                    current_mtime = xml_path.stat().st_mtime
                except FileNotFoundError:
                    current_mtime = None

                if current_mtime is not None and current_mtime != last_good_mtime:
                    try:
                        probe_model, _ = load_model(xml_path)
                    except Exception as exc:  # pragma: no cover - interactive tool
                        if last_failed_mtime != current_mtime:
                            print(f"Reload blocked by XML error: {exc}")
                            last_failed_mtime = current_mtime
                        handle.set_texts(
                            [
                                *overlay_text(xml_path),
                                (
                                    mujoco.mjtFontScale.mjFONTSCALE_100,
                                    mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                    "Reload failed",
                                    str(exc),
                                ),
                            ]
                        )
                    else:
                        del probe_model
                        break

                handle.sync()
                time.sleep(0.05)

        if not xml_path.exists():
            time.sleep(1.0)
            continue

        try:
            last_good_mtime = xml_path.stat().st_mtime
        except FileNotFoundError:
            last_good_mtime = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the running robot MJCF in the MuJoCo viewer.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to the MJCF file to load.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Regenerate the MJCF scaffold before opening the viewer.",
    )
    args = parser.parse_args()

    xml_path = args.xml.resolve()

    if args.build or not xml_path.exists():
        runpy.run_path(str(ROOT / "mujoco" / "build_debug_model.py"), run_name="__main__")

    run(xml_path)


if __name__ == "__main__":
    main()