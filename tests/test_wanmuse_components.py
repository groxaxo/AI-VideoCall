import base64
import time
import unittest

import numpy as np

from core.audio_pcm import decode_pcm16_base64
from core.wanmuse.frame_source import LatestFrameStore
from core.wanmuse.musetalk_client import MuseTalkClientError, parse_render_response
from core.wanmuse.settings import parse_face_bbox


class Pcm16DecoderTests(unittest.TestCase):
    def test_decodes_little_endian_pcm16(self):
        raw = np.array([-32768, 0, 16384, 32767], dtype="<i2").tobytes()
        decoded = decode_pcm16_base64(base64.b64encode(raw).decode("ascii"))
        np.testing.assert_allclose(
            decoded,
            np.array([-1.0, 0.0, 0.5, 32767 / 32768], dtype=np.float32),
        )

    def test_rejects_invalid_base64(self):
        with self.assertRaises(ValueError):
            decode_pcm16_base64("%%3")

    def test_rejects_odd_byte_count(self):
        value = base64.b64encode(b"abc").decode("ascii")
        with self.assertRaises(ValueError):
            decode_pcm16_base64(value)


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
                {"frames": [encoded, encoded], "fps": 25}, max_frames=1
            )


class FaceBoundingBoxTests(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(parse_face_bbox(""))

    def test_parses_bbox(self):
        self.assertEqual(parse_face_bbox("10,20,110,220"), (10, 20, 110, 220))

    def test_rejects_inverted_bbox(self):
        with self.assertRaises(ValueError):
            parse_face_bbox("10,20,5,220")


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, value):
        import json

        self.messages.append(json.loads(value))


class _FakeSegment:
    def __init__(self, frames, fps=25):
        self.frames = tuple(frames)
        self.fps = fps


class _FakeMuseTalkClient:
    def __init__(self, frames=None, error=None):
        self.frames = frames or []
        self.error = error
        self.calls = []

    async def render(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeSegment(self.frames, kwargs["fps"])


class WanMuseManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import asyncio

        from core.wanmuse.manager import WanMuseLipSyncManager

        self.idle_event = asyncio.Event()
        self.manager = WanMuseLipSyncManager(self.idle_event)
        self.manager.websocket = _FakeWebSocket()
        self.manager.frame_store.update(
            np.zeros((16, 24, 3), dtype=np.uint8)
        )

    async def test_audio_is_attached_only_to_first_rendered_frame(self):
        encoded = base64.b64encode(b"jpeg").decode("ascii")
        fake = _FakeMuseTalkClient([encoded, encoded])
        self.manager.musetalk = fake

        await self.manager.process_audio_chunk(
            "audio-base64",
            np.zeros((1, 16000), dtype=np.float32),
        )

        messages = self.manager.websocket.messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["audio"], "audio-base64")
        self.assertEqual(messages[1]["audio"], "")
        self.assertEqual(messages[0]["video_backend"], "ti2v5b_musetalk")
        self.assertTrue(self.idle_event.is_set())

    async def test_sidecar_failure_falls_back_when_not_strict(self):
        self.manager.strict = False
        self.manager.musetalk = _FakeMuseTalkClient(error=RuntimeError("offline"))

        await self.manager.process_audio_chunk(
            "audio-base64",
            np.zeros((1, 8000), dtype=np.float32),
        )

        messages = self.manager.websocket.messages
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["audio"], "audio-base64")
        self.assertTrue(messages[0]["image"])
        self.assertTrue(self.idle_event.is_set())

    async def test_sidecar_failure_raises_in_strict_mode(self):
        self.manager.strict = True
        self.manager.musetalk = _FakeMuseTalkClient(error=RuntimeError("offline"))

        with self.assertRaises(RuntimeError):
            await self.manager.process_audio_chunk(
                "audio-base64",
                np.zeros((1, 8000), dtype=np.float32),
            )
        self.assertTrue(self.idle_event.is_set())


if __name__ == "__main__":
    unittest.main()
