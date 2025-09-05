from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any, BinaryIO, List


class UnsupportedReplayFormat(Exception):
    """Raised when a replay file does not match the expected toy format."""


class ReplayParser:
    """Parse AoE2DE replay files into structured events.

    The parser understands a very small subset of the replay format used for
    testing purposes. Each replay file starts with a 32-bit little-endian
    integer indicating the number of events. Every event then stores:

    * timestamp -- 32-bit little-endian integer of milliseconds
    * player name length (1 byte) followed by UTF-8 bytes
    * event name length (1 byte) followed by UTF-8 bytes

    Files with the ``.mgz`` extension are assumed to be gzip compressed while
    ``.aoe2record`` files are read as raw binary.
    """

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """Return a list of events extracted from ``file_path``.

        Parameters
        ----------
        file_path:
            Path to a ``.aoe2record`` or ``.mgz`` file.
        """

        with self._open(file_path) as fh:
            return self._parse_stream(fh)

    # ------------------------------------------------------------------
    def _open(self, path: Path) -> BinaryIO:
        if path.suffix.lower() == ".mgz":
            return gzip.open(path, "rb")
        return open(path, "rb")

    # ------------------------------------------------------------------
    def _parse_stream(self, fh: BinaryIO) -> List[dict[str, Any]]:
        # Read event count from header
        data = fh.read(4)
        if len(data) < 4:
            raise ValueError("File is too short to contain event count")
        (count,) = struct.unpack("<I", data)

        # Sanity-check the count. We require at least 6 bytes per event
        # (timestamp + two length bytes). If the declared number of events
        # cannot possibly fit in the remaining file, treat the replay as an
        # unsupported format.
        try:
            current = fh.tell()
            fh.seek(0, 2)
            end = fh.tell()
            fh.seek(current)
            remaining = end - current
            if count * 6 > remaining:
                raise UnsupportedReplayFormat(
                    "Unrealistic event count in header; unsupported replay format"
                )
        except OSError:
            # If we cannot seek, skip the validation and rely on decoding checks
            pass

        events: List[dict[str, Any]] = []
        for _ in range(count):
            ts_bytes = fh.read(4)
            if len(ts_bytes) < 4:
                raise ValueError("Corrupted replay: missing timestamp")
            (timestamp,) = struct.unpack("<I", ts_bytes)

            name_len_bytes = fh.read(1)
            if len(name_len_bytes) < 1:
                raise ValueError("Corrupted replay: missing player length")
            name_len = name_len_bytes[0]
            name_bytes = fh.read(name_len)
            if len(name_bytes) < name_len:
                raise ValueError("Corrupted replay: incomplete player name")
            player = self._decode_text(name_bytes)

            event_len_bytes = fh.read(1)
            if len(event_len_bytes) < 1:
                raise ValueError("Corrupted replay: missing event length")
            event_len = event_len_bytes[0]
            event_bytes = fh.read(event_len)
            if len(event_bytes) < event_len:
                raise ValueError("Corrupted replay: incomplete event name")
            event = self._decode_text(event_bytes)

            events.append({"timestamp": timestamp, "player": player, "event": event})

        return events

    # ------------------------------------------------------------------
    @staticmethod
    def _decode_text(data: bytes) -> str:
        """Decode bytes to text using UTF-8 with fallbacks.

        The toy format primarily uses UTF-8. Some replays may contain arbitrary
        byte sequences; in that case we fall back to Latin-1 or replace invalid
        characters so that the parser never raises ``UnicodeDecodeError``.
        """

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return data.decode("latin-1")
            except UnicodeDecodeError:
                return data.decode("utf-8", errors="replace")
