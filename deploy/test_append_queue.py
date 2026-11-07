"""Synthetic logic tests; not evidence of robot safety or RTC model alignment."""
import unittest
from append_queue import AppendQueue


class TestQueue(unittest.TestCase):
    def seeded(self):
        q = AppendQueue()
        token, prefix = q.begin()
        self.assertFalse(prefix)
        q.finish(token, 0, list(range(10)), list(range(10)))
        return q

    def test_result_early_preserves_old_tail(self):
        q = self.seeded()
        token, prefix = q.begin()
        self.assertEqual(q.pop(), 0)
        q.finish(token, len(prefix), list(range(100, 120)), list(range(100, 120)))
        self.assertEqual([q.pop() for _ in range(19)], list(range(1, 10)) + list(range(110, 120)))

    def test_result_late_does_not_count_idle_as_actions(self):
        q = self.seeded()
        token, prefix = q.begin()
        self.assertEqual([q.pop() for _ in range(10)], list(range(10)))
        self.assertIsNone(q.pop())
        self.assertIsNone(q.pop())
        q.finish(token, len(prefix), list(range(100, 120)), list(range(100, 120)))
        self.assertEqual(q.pop(), 110)

    def test_only_one_and_stale(self):
        q = self.seeded()
        token, prefix = q.begin()
        with self.assertRaises(RuntimeError): q.begin()
        q.finish(token, len(prefix), list(range(20)), list(range(20)))
        with self.assertRaises(RuntimeError): q.finish(token, 0, [], [])

    def test_timeout_invalidates_result_but_keeps_old_actions(self):
        q = self.seeded()
        token, prefix = q.begin()
        q.abort(token)
        self.assertEqual(q.pop(), 0)
        with self.assertRaises(RuntimeError): q.finish(token, len(prefix), [], [])
        new_token, _ = q.begin()
        self.assertNotEqual(token, new_token)


if __name__ == '__main__': unittest.main()
