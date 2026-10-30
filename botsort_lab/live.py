"""Mode 1: play with the tracker on the device camera (and record raw clips)."""
import time

import cv2

from .config import ARMS, make_cfgs
from .tracking import BotSortTracker


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            "Could not open the camera. On macOS, allow camera access for your terminal in "
            "System Settings > Privacy & Security > Camera, then restart the terminal."
        )
    return cap


def run_live(args):
    cfgs = make_cfgs()
    arm_keys = list(ARMS)
    cur = arm_keys.index(args.arm)
    tracker = BotSortTracker(args.weights, args.device, args.imgsz, args.classes or [0], args.det_conf)
    tracker.set_config(cfgs[arm_keys[cur]])
    cap = open_camera(args.camera)
    fps_ema, seen_ids = None, set()

    print("Keys: g = toggle GMC (resets tracker), r = reset tracker, q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        t0 = time.perf_counter()
        res, _, ids, _ = tracker.update(frame)
        dt = time.perf_counter() - t0
        fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 * (1 / dt)
        seen_ids.update(ids.tolist())

        vis = res.plot()
        cv2.putText(vis, f"{ARMS[arm_keys[cur]]['label']}  {fps_ema:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(vis, f"active tracks: {len(ids)}  unique IDs so far: {len(seen_ids)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("BoT-SORT live", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            cur = (cur + 1) % len(arm_keys)
            tracker.set_config(cfgs[arm_keys[cur]])
            seen_ids.clear()
            print("Switched to", ARMS[arm_keys[cur]]["label"])
        if key == ord("r"):
            tracker.reset()
            seen_ids.clear()
            print("Tracker reset")

    cap.release()
    cv2.destroyAllWindows()


def run_record(args):
    """Record a raw clip (no tracking) so frames arrive at the camera's true rate."""
    cap = open_camera(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera opened but returned no frames.")
    h, w = frame.shape[:2]
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    cam_fps = cam_fps if cam_fps and cam_fps > 1 else 30.0
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), cam_fps, (w, h))

    print(f"Recording {args.seconds}s at ~{cam_fps:.0f} FPS. q to stop early.")
    start, n = time.time(), 0
    while time.time() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    elapsed = time.time() - start
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {n} frames to {args.out} (measured {n / elapsed:.1f} FPS).")
