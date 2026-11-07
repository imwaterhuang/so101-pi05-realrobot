"""Asynchronous joint trace. Feedback errors use the latest command at read time."""
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path


class JointTrace:
    def __init__(self, directory, metadata):
        self.started = time.monotonic()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rtc_joints_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"
        self.file = self.path.open("x")
        self.events = queue.Queue(maxsize=10000)
        self.dropped = 0
        self.error = None
        self.closed = False
        self.file.write(json.dumps({"type": "metadata", "wall_time": datetime.now().astimezone().isoformat(),
                                    **metadata}) + "\n")
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def record(self, kind, **fields):
        if self.closed or self.error:
            return
        try:
            self.events.put_nowait({"type": kind, "t": time.monotonic() - self.started, **fields})
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            while True:
                row = self.events.get()
                if row is None:
                    break
                self.file.write(json.dumps(row, allow_nan=False) + "\n")
                self.file.flush()
        except Exception as exc:
            self.error = repr(exc)
        finally:
            self.file.close()

    def close(self):
        self.record("end", dropped=self.dropped)
        self.closed = True
        if self.worker.is_alive():
            try:
                self.events.put(None, timeout=2)
            except queue.Full:
                self.error = "Trace writer stalled"
            self.worker.join(timeout=5)
        print(f"Joint trace: {self.path}; dropped={self.dropped}; "
              f"error={self.error}; writer_alive={self.worker.is_alive()}", flush=True)
