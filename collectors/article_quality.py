"""Lightweight checks against server error pages masquerading as news.

Only Python standard library; safe in the Windows collection process.
"""
from __future__ import annotations

import re


_ERROR_HINTS = (
    "error establishing a database connection",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "please enable javascript to view this page",
    "cloudflare ray id",
    "the requested url was rejected",
    "доступ к сайту временно ограничен",
    "проверка браузера",
)


def is_error_page_text(text: str | None) -> bool:
    cleaned = re.sub(r"\s+", " ", text or "").strip().casefold()
    if not cleaned:
        return True
    if len(cleaned) > 1500:
        return False
    if any(phrase in cleaned for phrase in _ERROR_HINTS):
        return True
    if len(cleaned) < 350 and (
        cleaned.startswith(("access denied", "403 forbidden", "404 not found",
                            "502 bad gateway", "503 service unavailable",
                            "just a moment", "service unavailable",
                            "ошибка 403", "ошибка 404"))
        or cleaned in ("forbidden", "not found", "captcha", "доступ запрещён")
    ):
        return True
    return False
