"""Experimental queue for offline tests only; not a robot controller."""
from collections import deque
from threading import Lock


class AppendQueue:
    def __init__(self):
        self._lock = Lock()
        self._pending = deque()
        self._generation = 0
        self._inflight = False

    def begin(self, max_prefix=32):
        with self._lock:
            if self._inflight:
                raise RuntimeError('a request is already pending')
            if len(self._pending) > max_prefix:
                raise ValueError('queue exceeds available RTC prefix')
            self._inflight = True
            return self._generation, tuple(raw for raw, _ in self._pending)

    def pop(self):
        with self._lock:
            if not self._pending:
                return None
            return self._pending.popleft()[1]

    def finish(self, token, prefix_length, raw, processed, *, replace=False):
        with self._lock:
            if not self._inflight or token != self._generation:
                raise RuntimeError('stale or unsolicited result')
            if len(raw) != len(processed) or not 0 <= prefix_length <= len(raw):
                raise ValueError('invalid action chunk or prefix')
            if replace:
                # Only the prefix steps consumed since begin() have elapsed.
                prefix_length -= len(self._pending)
                if prefix_length < 0:
                    raise ValueError("prefix shorter than pending queue")
                self._pending.clear()
            previous_tail = self._pending[-1][1] if self._pending else None
            incoming_first = processed[prefix_length] if prefix_length < len(processed) else None
            self._pending.extend(zip(raw[prefix_length:], processed[prefix_length:]))
            self._generation += 1
            self._inflight = False
            return previous_tail, incoming_first

    def size(self):
        with self._lock:
            return len(self._pending)

    def abort(self, token):
        """Invalidate an outstanding request while preserving its old queue."""
        with self._lock:
            if not self._inflight or token != self._generation:
                raise RuntimeError('stale or unsolicited abort')
            self._generation += 1
            self._inflight = False
