from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import zmq

logger = logging.getLogger("ti2v_frame_publisher")


class LatestFramePublisher:
    """Publish encoded TI2V frames using the WanMuse latest-frame contract."""

    def __init__(
        self, endpoint: str, *, topic: str = "", jpeg_quality: int = 92
    ):
        self.endpoint = endpoint
        self.topic = topic.encode("utf-8")
        self.jpeg_quality = min(100, max(40, int(jpeg_quality)))
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.bind(endpoint)

    def publish(self, frame) -> bool:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode frame")
        payload = encoded.tobytes()
        try:
            if self.topic:
                self.socket.send_multipart(
                    [self.topic, payload], flags=zmq.NOBLOCK
                )
            else:
                self.socket.send(payload, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            # A slow or absent subscriber must never block the TI2V renderer.
            return False

    def close(self) -> None:
        self.socket.close(linger=0)


def publish_video(
    path: Path, endpoint: str, topic: str, realtime: bool
) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    publisher = LatestFramePublisher(endpoint, topic=topic)
    try:
        # PUB/SUB subscribers need a short subscription warm-up interval.
        time.sleep(0.3)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            publisher.publish(frame)
            if realtime:
                time.sleep(1.0 / fps)
    finally:
        capture.release()
        publisher.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a video as WanMuse TI2V latest-frame messages"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--topic", default="")
    parser.add_argument("--no-realtime", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    publish_video(
        args.video,
        args.endpoint,
        args.topic,
        not args.no_realtime,
    )


if __name__ == "__main__":
    main()
