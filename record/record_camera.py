"""Record a camera (default: iPhone desk_view, index 2) to MP4 until Ctrl+C.

Runs as its own process next to rtc_real_client.py, so the control loop is untouched.
Each frame gets a wall-clock overlay, and a CSV of wall-clock timestamps is written
next to the video, so frames can be matched to the joint trace afterwards.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def fit(frame, w, h):
    """Scale to fit inside w x h keeping aspect ratio, pad the rest black."""
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = max(1, round(fw * s)), max(1, round(fh * s))
    out = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=int, default=2, help='OpenCV camera index (2 = iPhone desk_view)')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--out', default=None, help='Output .mp4 (default logs/videos/task_<time>.mp4)')
    p.add_argument('--no-overlay', action='store_true', help='Do not draw the timestamp on frames')
    p.add_argument('--seconds', type=float, default=0, help='Stop after N seconds (0 = until Ctrl+C)')
    args = p.parse_args()

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / 'logs' / 'videos' / f'task_{datetime.now():%Y%m%d_%H%M%S}.mp4')
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit(f'Camera {args.index} unavailable (camera permission? iPhone unlocked and nearby?)')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = False, None
    for _ in range(50):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.1)
    if not ok:
        raise SystemExit('Camera opened but returned no frames')
    # The output size is fixed. When rtc_real_client.py opens the same iPhone it switches the
    # camera format (e.g. 640x360 -> 640x480), and the writer silently drops frames whose size
    # differs, so every frame is letterboxed to this size.
    w, h = args.width, args.height

    writer = None
    # mp4v (MPEG-4 Part 2): the bundled FFmpeg's H.264 path opens but then drops frames.
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit('Could not open a video writer')

    stamps = open(out.with_suffix('.csv'), 'w', newline='')
    log = csv.writer(stamps)
    log.writerow(['frame', 'video_frame', 'wall_time', 'iso_time'])
    print(f'RECORDING camera {args.index} {w}x{h} -> {out}  (Ctrl+C to stop)', flush=True)

    # The camera delivers ~27 fps, not exactly --fps. Frames are repeated as needed so the
    # file stays at a constant --fps and plays back at real speed.
    n, written, start, last_shape = 0, 0, time.monotonic(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if frame.shape[:2] != (h, w):
                if frame.shape[:2] != last_shape:
                    print(f'  camera format now {frame.shape[1]}x{frame.shape[0]}; fitting into {w}x{h}', flush=True)
                    last_shape = frame.shape[:2]
                frame = fit(frame, w, h)
            now = time.time()
            iso =datetime.fromtimestamp(now).strftime('%H:%M:%S.%f')[:-3]
            if not args.no_overlay:
                cv2.putText(frame, iso, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, iso, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            log.writerow([n, written, f'{now:.3f}', iso])
            target = max(written + 1, int((time.monotonic() - start) * args.fps))
            while written < target:
                writer.write(frame)
                written += 1
            n += 1
            if n % 300 == 0:
                print(f'  {n} frames, {n / (time.monotonic() - start):.1f} fps', flush=True)
            if args.seconds and time.monotonic() - start >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - start
        writer.release()
        cap.release()
        stamps.close()
        fps = n / elapsed if elapsed else 0
        print(f'SAVED {out}  {elapsed:.1f}s, camera {fps:.1f} fps, {written} frames written at {args.fps:g} fps',
              flush=True)


if __name__ == '__main__':
    main()
