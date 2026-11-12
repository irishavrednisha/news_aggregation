"""Offline image extraction/download tests; no network, PostgreSQL, or MAX."""
import os
import tempfile
import unittest
import zlib
import struct
from pathlib import Path
from unittest.mock import patch

from bs4 import BeautifulSoup

from collectors.image_extractor import ImageCandidate, extract_image_candidates
from processing.image_downloader import _assert_public_http, _image_metadata, download_main_image


def _png(width=600, height=400):
    """Create a small, valid RGB PNG using Python standard library."""
    def chunk(name, data):
        return (struct.pack(">I", len(data)) + name + data
                + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(
        b"\x00" + b"\xff\x00\x00" * width for _ in range(height)
    )
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


class _Response:
    def __init__(self, body):
        self.body = body
        self.status_code = 200
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset + chunk_size]


class ImagePipelineTests(unittest.TestCase):
    def test_og_image_prioritized_and_relative_urls_resolved(self):
        html = """<html><head>
          <meta property="og:image" content="/media/main.jpg">
          </head><body><main><figure><img data-src="/photo.jpg">
          <figcaption>Фото пресс-службы</figcaption></figure>
          <img src="/assets/logo.svg"><img src="/ads/banner.jpg">
          </main></body></html>"""
        images = extract_image_candidates(
            BeautifulSoup(html, "html.parser"),
            "https://example.org/news/123",
        )
        self.assertEqual(images[0].url, "https://example.org/media/main.jpg")
        self.assertEqual(images[1].url, "https://example.org/photo.jpg")
        self.assertIn("пресс-службы", images[1].caption)
        self.assertNotIn("banner", " ".join(x.url for x in images))

    def test_responsive_large_image_and_no_outside_sidebar(self):
        html = """<aside><img src="/unrelated.jpg"></aside>
          <article><img src="data:image/png;base64,abcd"
          srcset="/small.jpg 120w, /large.jpg 1200w"></article>"""
        images = extract_image_candidates(
            BeautifulSoup(html, "html.parser"), "https://example.org/post"
        )
        self.assertEqual([x.url for x in images], ["https://example.org/large.jpg"])

    def test_silicon_name_not_accidentally_filtered(self):
        html = '<meta property="og:image" content="/news/silicon.jpg">'
        images = extract_image_candidates(
            BeautifulSoup(html, "html.parser"), "https://example.org/news/1"
        )
        self.assertEqual(len(images), 1)

    def test_format_dimensions_and_small_image_rejection(self):
        valid = _png()
        self.assertEqual(_image_metadata(valid), ("image/png", ".png", 600, 400))
        with self.assertRaisesRegex(ValueError, "too small"):
            _image_metadata(_png(100, 100))
        with self.assertRaises(ValueError):
            _image_metadata(b"not an image")

    def test_private_address_is_rejected(self):
        with patch("processing.image_downloader.socket.getaddrinfo",
                   return_value=[(2, 1, 6, "", ("127.0.0.1", 443))]):
            with self.assertRaisesRegex(ValueError, "non-public"):
                _assert_public_http("https://example.org/image.png")

    def test_download_one_main_image_and_keep_unknown_rights(self):
        image = _png()
        with tempfile.TemporaryDirectory() as directory:
            with patch("processing.image_downloader.socket.getaddrinfo",
                       return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
                with patch("processing.image_downloader.requests.get",
                           return_value=_Response(image)) as request:
                    result = download_main_image(
                        [ImageCandidate("https://example.org/industrial.png")],
                        storage_dir=directory,
                    )
            self.assertIsNotNone(result)
            self.assertTrue(Path(result.local_path).is_file())
            self.assertEqual(Path(result.local_path).read_bytes(), image)
            self.assertEqual(result.width, 600)
            self.assertEqual(result.mime_type, "image/png")
            self.assertEqual(request.call_count, 1)
            self.assertTrue(request.call_args.kwargs["verify"])
            self.assertFalse(request.call_args.kwargs["allow_redirects"])


if __name__ == "__main__":
    unittest.main()
