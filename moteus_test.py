import argparse
import asyncio
import math
import time
import moteus


async def spin(duration, velocity, maximum_torque, watchdog_timeout, delay,
               kp_scale=0.0, ignore_position_bounds=False, debug=False):
    c = moteus.Controller(id=1, transport=moteus.Fdcanusb(path="COM10"))
    start = time.time()
    i = 0
    try:
        result = await c.set_stop(query=True)
        print(result)
        while time.time() - start < duration:
            # periodically request a query for diagnostics when debug is enabled
            do_query = debug and (i % max(1, int(0.5 / delay)) == 0)
            res = await c.set_position(
                position=math.nan,
                velocity=velocity,
                maximum_torque=maximum_torque,
                watchdog_timeout=watchdog_timeout,
                kp_scale=kp_scale,
                ignore_position_bounds=1 if ignore_position_bounds else None,
                query=do_query,
            )
            if do_query:
                print(res)
            await asyncio.sleep(delay)
            i += 1
    finally:
        # Stop cleanly: command zero velocity and low torque
        q = await c.set_position(
            position=math.nan,
            velocity=0.0,
            maximum_torque=0.0,
            watchdog_timeout=1.0,
            query=True,
        )
        print(q)


async def move_to_position(position, maximum_torque, velocity=None, watchdog_timeout=0.5,
                           update_period=0.02, ignore_position_bounds=False,
                           debug=False, kp_scale=None, kd_scale=None):
    """Move to a specific position and hold."""
    c = moteus.Controller(id=1, transport=moteus.Fdcanusb(path="COM10"))
    try:
        result = await c.set_stop(query=True)
        print(result)
        # Send position command with optional velocity limit
        res = await c.set_position(
            position=position,
            velocity=velocity,
            maximum_torque=maximum_torque,
            watchdog_timeout=watchdog_timeout,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            ignore_position_bounds=1 if ignore_position_bounds else None,
            query=True,
        )
        print(f"Target position: {position} rev")
        print(res)
        
        # Check position periodically until we reach it
        while True:
            await asyncio.sleep(update_period)
            try:
                res = await c.set_position(
                    position=position,
                    velocity=velocity,
                    maximum_torque=maximum_torque,
                    watchdog_timeout=watchdog_timeout,
                    kp_scale=kp_scale,
                    kd_scale=kd_scale,
                    ignore_position_bounds=1 if ignore_position_bounds else None,
                    query=True,
                )
                # Always print status
                print(res)
                # Stop when position is close enough (within 0.01 rev tolerance)
                actual_pos = res.values[moteus.Register.POSITION]
                if abs(actual_pos - position) < 0.01:
                    print(f"Reached position: {actual_pos} rev")
                    # break
            except Exception as e:
                print(f"Error during movement: {e}")
                raise
    except Exception as e:
        print(f"Exception in move_to_position: {e}")
        raise
    finally:
        # Stop cleanly
        try:
            q = await c.set_position(
                position=math.nan,
                velocity=0.0,
                maximum_torque=0.0,
                watchdog_timeout=1.0,
                query=True,
            )
            print(q)
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    p = argparse.ArgumentParser(description="Moteus motor control")
    
    # Position mode arguments
    p.add_argument("--position", type=float, default=None, help="target position (rev) - enables position mode")
    p.add_argument("--velocity-limit", type=float, default=None, help="max velocity during position move (rev/s)")
    p.add_argument("--pos-kp-scale", type=float, default=None, help="position mode: proportional gain scale multiplier")
    p.add_argument("--pos-kd-scale", type=float, default=None, help="position mode: derivative gain scale multiplier")
    p.add_argument("--pos-watchdog", type=float, default=0.5, help="position mode watchdog timeout (s)")
    p.add_argument("--pos-period", type=float, default=0.02, help="position mode command interval (s)")
    
    # Velocity mode arguments (legacy)
    p.add_argument("--duration", type=float, default=5.0, help="seconds to spin (velocity mode)")
    p.add_argument("--velocity", type=float, default=2.0, help="rev/s, + or - (velocity mode)")
    p.add_argument("--delay", type=float, default=0.05, help="command interval (s, velocity mode)")
    p.add_argument("--kp-scale", type=float, default=0.0, help="kp_scale (0 for pure velocity)")
    p.add_argument("--watchdog", type=float, default=0.25, help="watchdog timeout (s)")
    
    # Common arguments
    p.add_argument("--max-torque", type=float, default=3.0, help="maximum torque (N*m)")
    p.add_argument("--ignore-bounds", action="store_true", help="ignore configured position bounds")
    p.add_argument("--debug", action="store_true", help="print occasional query responses")
    args = p.parse_args()

    if args.position is not None:
        # Position mode
        asyncio.run(move_to_position(
            position=args.position,
            maximum_torque=args.max_torque,
            velocity=args.velocity_limit,
            watchdog_timeout=args.pos_watchdog,
            update_period=args.pos_period,
            ignore_position_bounds=args.ignore_bounds,
            debug=args.debug,
            kp_scale=args.pos_kp_scale,
            kd_scale=args.pos_kd_scale
        ))
    else:
        # Velocity mode (original)
        asyncio.run(spin(
            args.duration, args.velocity, args.max_torque, args.watchdog, args.delay,
            kp_scale=args.kp_scale, ignore_position_bounds=args.ignore_bounds,
            debug=args.debug
        ))


if __name__ == "__main__":
    main()