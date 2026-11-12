"""Download ONE validated main news image; no Pillow or native extensions.

Safety: no TLS bypass, refuse local/private network destinations and redirects,
cap network bytes and pixels, validate magic bytes, use atomic file writes.
Only metadata is stored in PostgreSQL. Download does not grant publishing rights.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from collectors.image_extractor import ImageCandidate


MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000
MIN_WIDTH = 350
MIN_HEIGHT = 180
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsAggregator/1.0)",
           "Accept": "image/avif,image/webp,image/png,image/jpeg;q=0.9,*/*;q=0.2"}


@dataclass(frozen=True)
class DownloadedImage:
    source_url: str
    local_path: str
    mime_type: str
    width: int
    height: int
    sha256: str
    caption: str = ""
    credit: str = ""
    license_url: str = ""


def _assert_public_http(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only HTTP(S) image URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("Image URL cannot contain credentials")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Unexpected image URL port")
    # Reject private, loopback, link-local, reserved and mixed DNS responses.
    infos = socket.getaddrinfo(parsed.hostname, parsed.port or
                              (443 if parsed.scheme == "https" else 80),
                              type=socket.SOCK_STREAM)
    if not infos or any(not ipaddress.ip_address(i[4][0]).is_global for i in infos):
        raise ValueError("Image URL resolves to non-public address")


def _jpeg_size(data: bytes) -> tuple[int, int]:
    pos = 2
    while pos + 9 < len(data):
        if data[pos] != 0xff:
            pos += 1
            continue
        marker = data[pos + 1]
        pos += 2
        if marker in (0xD8, 0xD9, 0x01) or (0xD0 <= marker <= 0xD7):
            continue
        if pos + 2 > len(data):
            break
        length = struct.unpack_from(">H", data, pos)[0]
        if length < 2 or pos + length > len(data):
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if length < 7:
                break
            height, width = struct.unpack_from(">HH", data, pos + 3)
            return width, height
        pos += length
    raise ValueError("Cannot read JPEG image dimensions")


def _image_metadata(data: bytes) -> tuple[str, str, int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack_from(">II", data, 16)
        mime, ext = "image/png", ".png"
    elif data.startswith(b"\xff\xd8\xff"):
        width, height = _jpeg_size(data)
        mime, ext = "image/jpeg", ".jpg"
    elif len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        typ = data[12:16]
        if typ == b"VP8X":
            width = 1 + int.from_bytes(data[24:27], "little")
            height = 1 + int.from_bytes(data[27:30], "little")
        elif typ == b"VP8L" and data[20] == 0x2F:
            bits = int.from_bytes(data[21:25], "little")
            width = 1 + (bits & 0x3FFF)
            height = 1 + ((bits >> 14) & 0x3FFF)
        elif typ == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            width = struct.unpack_from("<H", data, 26)[0] & 0x3FFF
            height = struct.unpack_from("<H", data, 28)[0] & 0x3FFF
        else:
            raise ValueError("Unrecognized WebP encoding")
        mime, ext = "image/webp", ".webp"
    else:
        raise ValueError("Image format is unsupported (JPEG, PNG, WebP only)")

    if not (MIN_WIDTH <= width <= 15000 and MIN_HEIGHT <= height <= 15000):
        raise ValueError("Image is too small or too large")
    if width * height > MAX_PIXELS or not (0.35 <= width / height <= 5.0):
        raise ValueError("Image dimensions or aspect ratio are unsuitable")
    return mime, ext, width, height


def _verify_tls():
    ca_bundle = os.environ.get("NEWS_CA_BUNDLE", "").strip()
    if ca_bundle:
        if not Path(ca_bundle).is_file():
            raise ValueError("NEWS_CA_BUNDLE points to missing file")
        return ca_bundle
    return True


def download_main_image(
    candidates: list[ImageCandidate],
    storage_dir: str | Path = "data/news_images",
    max_candidates: int = 3,
) -> DownloadedImage | None:
    """Try candidates in order. Return first valid image or None.

    No exceptions from individual candidate should interrupt news collection.
    DNS/TLS failures are logged without leaking secrets and never bypassed.
    """
    directory = Path(storage_dir)
    for candidate in candidates[:max_candidates]:
        current_url = candidate.url
        try:
            for _ in range(4):
                _assert_public_http(current_url)
                with requests.get(
                    current_url, stream=True, timeout=(8, 20),
                    headers=HEADERS, verify=_verify_tls(),
                    allow_redirects=False,
                ) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location", "")
                        if not location:
                            raise ValueError("Image redirect without Location")
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    declared = response.headers.get("Content-Length")
                    if declared and int(declared) > MAX_BYTES:
                        raise ValueError("Image exceeds download size limit")
                    data = bytearray()
                    for chunk in response.iter_content(chunk_size=65536):
                        data.extend(chunk)
                        if len(data) > MAX_BYTES:
                            raise ValueError("Image exceeds download size limit")
                    mime, extension, width, height = _image_metadata(bytes(data))
                    digest = hashlib.sha256(data).hexdigest()
                    directory.mkdir(parents=True, exist_ok=True)
                    destination = directory / (digest + extension)
                    if not destination.exists():
                        tmpname = None
                        try:
                            with tempfile.NamedTemporaryFile(
                                mode="wb", dir=directory, prefix=".img_",
                                suffix=".part", delete=False
                            ) as tmp:
                                tmpname = tmp.name
                                tmp.write(data)
                            os.replace(tmpname, destination)
                        finally:
                            if tmpname and os.path.exists(tmpname):
                                os.unlink(tmpname)
                    return DownloadedImage(
                        source_url=candidate.url,
                        local_path=str(destination),
                        mime_type=mime,
                        width=width,
                        height=height,
                        sha256=digest,
                        caption=candidate.caption,
                        credit=candidate.credit,
                        license_url=candidate.license_url,
                    )
            raise ValueError("Too many redirects")
        except (ValueError, OSError, requests.RequestException) as exc:
            print(f"Пропущено изображение {candidate.url[:120]}: {type(exc).__name__}: {exc}")
    return None
