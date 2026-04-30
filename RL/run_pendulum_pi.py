"""Run an exported ONNX pendulum policy on the Raspberry Pi with moteus.

This script reads the live controller telemetry, feeds the policy with raw
hardware observations [position_turns, velocity_turns_per_s, torque_nm], and
writes a torque command back to the controller at 50 Hz.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import moteus


class OnnxPolicy:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, obs):
        if obs.ndim == 1:
            obs = obs[None, :]
        action = self.session.run([self.output_name], {self.input_name: obs.astype(np.float32)})[0]
        return float(np.clip(action.reshape(-1)[0], -1.0, 1.0))


async def run(model_path: Path, device: str, rate_hz: float, max_torque: float, watchdog_timeout: float, debug: bool):
    policy = OnnxPolicy(model_path)
    # Use the default Controller() which auto-selects available transports.
    # Creating a transport (Fdcanusb) and passing it in has triggered
    # C++ assertion failures on some systems; the default constructor
    # has proven more robust in practice (see quick tests earlier).
    try:
        controller = moteus.Controller()
        if debug:
            print(f"Controller initialized: {controller}")
    except Exception as e:
        print(f"Failed to initialize moteus.Controller(): {e}")
        raise

    period_s = 1.0 / rate_hz

    def obs_from_result(result):
        values = result.values
        return np.array([
            float(values[moteus.Register.POSITION]),
            float(values[moteus.Register.VELOCITY]),
            float(values[moteus.Register.TORQUE]),
        ], dtype=np.float32)

    result = await controller.set_stop(query=True)
    obs = obs_from_result(result)

    try:
        while True:
            start = time.monotonic()
            torque = policy.predict(obs)
            torque = float(np.clip(torque, -max_torque, max_torque))

            result = await controller.set_position(
                position=math.nan,
                velocity=math.nan,
                feedforward_torque=torque,
                maximum_torque=max_torque,
                watchdog_timeout=watchdog_timeout,
                kp_scale=0.0,
                kd_scale=0.0,
                ignore_position_bounds=1,
                query=True,
            )
            obs = obs_from_result(result)

            if debug:
                print(
                    f"pos={obs[0]: .3f} rev  vel={obs[1]: .3f} rev/s  "
                    f"torque={obs[2]: .3f} Nm  cmd={torque: .3f} Nm  "
                    f"mode={result.values.get(moteus.Register.MODE, 'n/a')}  "
                    f"fault={result.values.get(moteus.Register.FAULT, 'n/a')}"
                )

            elapsed = time.monotonic() - start
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
    finally:
        try:
            await controller.set_stop(query=True)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pendulum ONNX policy on moteus hardware")
    parser.add_argument("--model", required=True, help="Path to the exported .onnx policy")
    parser.add_argument("--device", default="/dev/ttyACM0", help="fdcanusb device path")
    parser.add_argument("--rate", type=float, default=50.0, help="Control rate in Hz")
    parser.add_argument("--max-torque", type=float, default=1.0, help="Maximum torque to command (Nm)")
    parser.add_argument("--watchdog", type=float, default=0.1, help="Watchdog timeout (s)")
    parser.add_argument("--debug", action="store_true", help="Print telemetry every cycle")
    args = parser.parse_args()

    asyncio.run(run(Path(args.model), args.device, args.rate, args.max_torque, args.watchdog, args.debug))
