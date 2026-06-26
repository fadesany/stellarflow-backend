from __future__ import annotations

from typing import Any, Generator
from src.serialization.encoders import unpack_bundle, FRAME_SIZE

class StreamBuffer:
    """Accumulate binary chunks and yield parsed TelemetryFrames."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes | bytearray | memoryview) -> Generator[Any, None, None]:
        """Append *data* and yield every complete binary telemetry frame.

        Frames are fixed-size (FRAME_SIZE bytes), allowing efficient slicing
        without scanning for delimiters.
        """
        self._buf += data

        # Calculate how many full frames we have
        num_frames = len(self._buf) // FRAME_SIZE
        if num_frames == 0:
            return

        # Extract the contiguous block of full frames
        consumed = num_frames * FRAME_SIZE
        frame_data = bytes(self._buf[:consumed])
        del self._buf[:consumed]

        # Use the binary unpacker to yield TelemetryFrame objects
        yield from unpack_bundle(frame_data)

    def reset(self) -> None:
        """Discard all buffered data."""
        self._buf.clear()

__all__ = ["StreamBuffer"]
