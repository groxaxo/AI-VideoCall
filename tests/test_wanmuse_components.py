import base64
import time
import unittest

import numpy as np

from core.wanmuse.frame_source import LatestFrameStore
from core.wanmuse.musetalk_client import (
    MuseTalkClientError,
    parse_render_response,
)
from core.wanmuse.settings import parse_face_bbox


class LatestFrameStoreTests(unittest.TestCase):
    def test_update_copies_input_and_snapshot_copies_output(self):
        store = LatestFrameStore()
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        sequence = store.update(frame)
        self.assertEqual(sequence, 1)

        frame[:] = 255
        snapshot = store.snapshot()
        self.assertIsNotNone(snapshot)
        self.assertTrue(np.all(snapshot.frame == 0))

        snapshot.frame[:] = 127
        second = store.snapshot()
        self.assertTrue(np.all(second.frame == 0))

    def test_stale_frame_is_not_returned(self):
        store = LatestFrameStore()
        store.update(
            np.zeros((2, 2, 3), dtype=np.uint8),
            received_at=time.monotonic() - 5.0,
        )
        self.assertIsNone(store.snapshot(max_age_seconds=1.0))
        self.assertIsNotNone(store.snapshot(max_age_seconds=None))

    def test_rejects_invalid_shape(self):
        store = LatestFrameStore()
        with self.assertRaises(ValueError):
            store.update(np.zeros((10, 10), dtype=np.uint8))


class MuseTalkResponseTests(unittest.TestCase):
    def test_valid_response(self):
        encoded = base64.b64encode(b"jpeg-bytes").decode("ascii")
        response = parse_render_response(
            {
                "request_id": "abc",
                "frames": [encoded, encoded],
                "fps": 25,
                "duration_seconds": 0.08,
            }
        )
        self.assertEqual(response.request_id, "abc")
        self.assertEqual(response.fps, 25)
        self.assertEqual(len(response.frames), 2)

    def test_rejects_invalid_base64(self):
        with self.assertRaises(MuseTalkClientError):
            parse_render_response({"frames": ["%%%"], "fps": 25})

    def test_rejects_excess_frames(self):
        encoded = base64.b64encode(b"x").decode("ascii")
        with self.assertRaises(MuseTalkClientError):
            parse_render_response(
                {"frames": [encoded, encoded], "fps": 25},
                max_frames=1,
            )


class FaceBoundingBoxTests(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(parse_face_bbox(""))

    def test_parses_bbox(self):
        self.assertEqual(
            parse_face_bbox("10,20,110,220"),
            (10, 20, 110, 220),
        )

    def test_rejects_inverted_bbox(self):
        with self.assertRaises(ValueError):
            parse_face_bbox("10,20,5,220")


if __name__ == "__main__":
    unittest.main()
