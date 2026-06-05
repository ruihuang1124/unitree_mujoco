#!/usr/bin/env python3
import argparse
import math
import socket
import time


def main():
    parser = argparse.ArgumentParser(description="Send a fake HLC debug pose to the MuJoCo visualizer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39001)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / max(args.rate, 1.0)
    end_time = time.time() + args.duration
    start_time = time.time()

    while time.time() < end_time:
        t = time.time() - start_time
        target = (0.65 + 0.05 * math.sin(t), 0.0, 0.65)
        ee = (0.55, 0.0, 0.58 + 0.03 * math.sin(2.0 * t))
        quat = (1.0, 0.0, 0.0, 0.0)
        progress = min(t / max(args.duration, 1.0e-6), 1.0)
        err = math.sqrt(sum((target[i] - ee[i]) ** 2 for i in range(3)))
        packet = (
            "HLCDBG 1 "
            f"{target[0]:.6f} {target[1]:.6f} {target[2]:.6f} "
            f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
            f"{ee[0]:.6f} {ee[1]:.6f} {ee[2]:.6f} "
            f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
            f"{progress:.6f} {err:.6f}"
        )
        sock.sendto(packet.encode("ascii"), (args.host, args.port))
        time.sleep(period)


if __name__ == "__main__":
    main()
