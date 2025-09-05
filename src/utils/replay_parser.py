"""Replay parser using the :mod:`mgz` library.

This module replaces the previous toy format implementation with a thin
wrapper around the `mgz` package which understands the real
``.aoe2record`` specification.  Only high level metadata is exposed at the
moment, but the parser can be extended to yield additional information if
needed.

The parser reads the replay header and returns a dictionary containing
game version, map information and player details.  During parsing the
checksum fields present in the header's string blocks are validated to
ensure file integrity.
"""

from __future__ import annotations

import gzip
import logging
import zlib
from pathlib import Path
from typing import Any, BinaryIO, Dict, List

import mgz
from construct import ConstructError


class UnsupportedReplayFormat(Exception):
    """Raised when a replay file does not conform to ``.aoe2record``."""


class ReplayParser:
    """Parse AoE2DE replay files into metadata dictionaries."""

    # ------------------------------------------------------------------
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Return metadata extracted from ``file_path``.

        Parameters
        ----------
        file_path:
            Path to a replay file.  Gzip-compressed files are detected by
            inspecting the file's magic number rather than relying on the
            extension.
        """

        with self._open(file_path) as fh:
            try:
                header = mgz.header.parse_stream(fh)
            except ConstructError as exc:  # pragma: no cover - defensive
                raise UnsupportedReplayFormat(
                    "File is not a valid aoe2record"
                ) from exc

        self._validate_checksums(header)
        return self._extract_metadata(header)

    # ------------------------------------------------------------------
    def _open(self, path: Path) -> BinaryIO:
        """Open ``path`` returning a binary file handle.

        The first few bytes of the file are inspected to detect gzip
        compression.  If the gzip magic number is present the file is opened
        with :func:`gzip.open`; otherwise the built-in :func:`open` is used.
        """

        raw = open(path, "rb")
        magic = raw.read(2)
        raw.seek(0)
        if magic == b"\x1f\x8b":
            raw.close()
            return gzip.open(path, "rb")
        return raw

    # ------------------------------------------------------------------
    def _extract_metadata(self, header: Dict[str, Any]) -> Dict[str, Any]:
        players: List[Dict[str, Any]] = []
        de = header.get("de", {})
        for player in de.get("players", []):
            players.append(
                {
                    "name": self._decode_de_string(player.get("name")),
                    "civ_id": player.get("civ_id"),
                    "color_id": player.get("color_id"),
                }
            )

        map_info = header.get("map_info", {})

        return {
            "game_version": header.get("version"),
            "map": {
                "size_x": map_info.get("size_x"),
                "size_y": map_info.get("size_y"),
            },
            "players": players,
        }

    # ------------------------------------------------------------------
    def _validate_checksums(self, header: Dict[str, Any]) -> None:
        """Validate string block checksums in the header.

        ``.aoe2record`` headers contain several blocks of strings preceded by
        CRC32 values.  The :mod:`mgz` library exposes these blocks as arrays of
        ``{"crc": int, "string": {"value": bytes}}`` structures.  We recompute
        the CRC32 of each string and raise :class:`ValueError` if any do not
        match their advertised checksum.
        """

        def _iter_blocks(de_section: Dict[str, Any]):
            if not de_section:
                return
            if "rms_strings" in de_section:
                yield de_section["rms_strings"]
            for block in de_section.get("other_strings", []):
                yield block

        skipped = 0

        for block in _iter_blocks(header.get("de", {})):
            for entry in block.get("strings", []):
                if not isinstance(entry, dict):
                    skipped += 1
                    continue
                crc = entry.get("crc")
                if crc in (None, 0):
                    continue
                string_entry = entry.get("string")
                if isinstance(string_entry, dict):
                    data = string_entry.get("value", b"") or b""
                else:
                    data = b""
                if zlib.crc32(data) & 0xFFFFFFFF != crc:
                    raise ValueError("Checksum mismatch in string block")

        if skipped:
            logging.debug("Skipped %d non-dict checksum entries", skipped)

    # ------------------------------------------------------------------
    @staticmethod
    def _decode_de_string(field: Any) -> str:
        if not field:
            return ""
        data = field.get("value") if isinstance(field, dict) else field
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:  # pragma: no cover - fallback
                return data.decode("latin-1", errors="replace")
        return str(data)
