"""Fail-closed moderation gate for MAX publications.

Only a complete, approved Post snapshot can be sent. A Cluster being "ready"
is NOT permission to publish. Telegram approval/audit will be added later.
"""
from __future__ import annotations

import os
from typing import Mapping, Any


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def max_publishing_enabled(env: Mapping[str, str] | None = None) -> bool:
    """MAX sending is disabled unless explicitly opted in."""
    settings = os.environ if env is None else env
    value = settings.get("MAX_PUBLISHING_ENABLED", "")
    return str(value or "").strip().casefold() in _TRUE_VALUES


def split_post_snapshot(post_text: str) -> tuple[str, str]:
    """The approved snapshot stores 'title\\n\\nbody' as one immutable unit."""
    text = str(post_text or "").strip()
    if "\n\n" not in text:
        raise ValueError("Approved post must contain a title, blank line, and body")
    title, body = text.split("\n\n", 1)
    if not title.strip() or not body.strip():
        raise ValueError("Approved post title and body cannot be empty")
    return title.strip(), body.strip()


def is_approved_post(post: Any) -> bool:
    """Eligibility check, not an approval action.

    Only the future authenticated moderation service may create approved
    snapshots. This function never promotes a draft or a ready cluster.
    """
    if getattr(post, "status", None) != "approved":
        return False
    if getattr(post, "cluster_id", None) is None:
        return False
    try:
        split_post_snapshot(getattr(post, "post_text", None))
    except ValueError:
        return False
    return True
