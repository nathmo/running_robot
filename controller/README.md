# DASH-01 robot (hardware side)

Everything about the physical robot, self-contained so this folder can move to its own repo:

| folder | what |
|---|---|
| `fixed_gait/` | hand-crafted in-air walking demo (sim + CAN/moteus streaming) and the Flask web control UI (`fixed_gait/webui/`) |
| `robotCADdescription/` | CAD exports (URDF + the MJCF_OPEN_MUJOCO_B MuJoCo export the sim model is built from) |
| `model/` | dash01.xml (simulation-ready model, meshes referenced from `robotCADdescription/`), ride-height LUT, reachability plotting |
| `tools/` | AK60/AKE90 motor bring-up + CAN scan + URDF patching utilities |
| `viewer/` | browser MJCF debug viewer |
| `IMU.md` | **measured** properties of the Sense HAT (B) IMU — ranges, resolution, noise, bias, latency, frames, and a ready-to-use sim2real sensor model |
| `requirements-rpi.txt` | Raspberry Pi runtime deps (CAN streaming) |
| `requirements-webui.txt` | web control UI deps |

Hardware summary: 6 actuated DOF (per leg: hip_roll abduction, cam + thigh driving the sagittal
4-bar with parallel knee, passive ankle spring). Motors: AK60-39 + AKE90-8 over CAN
(can0 = right, can1 = left, IDs 104/105/106). Inertial sensing: Waveshare Sense HAT (B) on the Pi's
I2C bus, mounted under the base — see [IMU.md](IMU.md).

RL / policy training lives in `../training/` (its own self-contained copy of the model), which
stays in the training repo when this folder moves out.
