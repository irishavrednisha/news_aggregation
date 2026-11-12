"""Database-backed, fail-closed Telegram moderation (no Telegram dependency).

All human decisions require an authenticated moderator_id from the bot layer.
Post text and image selection are bound to a SHA-256 fingerprint. The MAX
publisher must re-check this fingerprint and the moderation audit.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from database.models import Cluster, News, Post
from database.moderation_models import ModerationAction, ModerationItem, NewsImage
from publishing.publisher import build_post_parts


def snapshot_hash(post_text: str, image_id: int | None) -> str:
    encoded = (post_text + "\0" + (str(image_id) if image_id is not None else "-")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _main_image(session: Session, cluster_id: int) -> NewsImage | None:
    return (
        session.query(NewsImage)
        .join(News, News.id == NewsImage.news_id)
        .filter(News.cluster_id == cluster_id)
        .order_by(News.is_primary.desc(), News.published_at.asc().nullslast(), NewsImage.id.asc())
        .first()
    )


def create_pending_items(session: Session, limit: int = 30) -> int:
    """Create ONE draft and moderation item per ready cluster.

    Run from one worker. Existing Post rows block repeat queue creation;
    never bypass a previously rejected/published decision by rerunning sync.
    """
    if limit <= 0:
        return 0
    clusters = (
        session.query(Cluster)
        .filter(Cluster.status == "ready", Cluster.summary_text.isnot(None))
        .order_by(Cluster.id.asc())
        .all()
    )
    count = 0
    for cluster in clusters:
        if count >= limit:
            break
        if not (cluster.tags or "").strip():
            continue
        if session.query(Post.id).filter(Post.cluster_id == cluster.id).first():
            continue
        title, body = build_post_parts(session, cluster)
        if not title.strip() or not body.strip():
            continue
        text = title.strip() + "\n\n" + body.strip()
        image = _main_image(session, cluster.id)
        post = Post(cluster_id=cluster.id, post_text=text, status="draft")
        session.add(post)
        session.flush()
        item = ModerationItem(
            post_id=post.id,
            status="pending",
            image_id=image.id if image else None,
            snapshot_sha256=snapshot_hash(text, image.id if image else None),
            image_rights_decision="pending" if image else "no_image",
        )
        session.add(item)
        session.commit()
        count += 1
    return count


def list_pending(session: Session, limit: int = 10):
    return (
        session.query(ModerationItem)
        .join(Post, Post.id == ModerationItem.post_id)
        .filter(ModerationItem.status == "pending", Post.status == "draft")
        .filter(ModerationItem.telegram_message_id.is_(None))
        .order_by(ModerationItem.id.asc())
        .limit(max(1, min(limit, 30)))
        .all()
    )


def _locked(session: Session, item_id: int):
    return (
        session.query(ModerationItem)
        .filter(ModerationItem.id == item_id)
        .with_for_update()
        .first()
    )


def get_item_preview(session: Session, item_id: int):
    item = session.get(ModerationItem, item_id)
    if item is None:
        return None
    post = session.get(Post, item.post_id)
    image = session.get(NewsImage, item.image_id) if item.image_id else None
    return item, post, image


def _audit(session: Session, item_id: int, moderator_id: int, action: str, details: dict | None = None):
    session.add(ModerationAction(
        moderation_item_id=item_id,
        moderator_telegram_id=moderator_id,
        action=action,
        details=details,
    ))


def decide(session: Session, item_id: int, action: str, moderator_id: int) -> str:
    """Apply one authenticated moderation action atomically.

    No action ever sends a MAX message.
    """
    if moderator_id <= 0:
        return "Недействительный Telegram ID."
    item = _locked(session, item_id)
    if item is None:
        return "Материал не найден."
    post = session.get(Post, item.post_id)
    if post is None:
        return "Публикация не найдена."
    if item.status != "pending" or post.status != "draft":
        return "Материал уже обработан. Повторное действие отменено."
    if item.snapshot_sha256 != snapshot_hash(post.post_text, item.image_id):
        return "Текст изменился после постановки в очередь. Одобрение заблокировано."

    if action == "remove_image":
        if item.image_id is None:
            return "У материала уже нет изображения."
        item.image_rights_decision = "remove_image"
        _audit(session, item_id, moderator_id, action)
        session.commit()
        return "Изображение исключено из будущей публикации. Теперь можно одобрить текст."

    if action == "rights_confirmed":
        if item.image_id is None:
            return "У материала нет изображения."
        item.image_rights_decision = "cleared"
        _audit(session, item_id, moderator_id, action, {
            "meaning": "Moderator attests independently verified republication permission."
        })
        session.commit()
        return "Отмечено подтверждение права на публикацию изображения. Теперь можно одобрить."

    if action == "reject":
        item.status = "rejected"
        item.moderator_telegram_id = moderator_id
        item.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        post.status = "rejected"
        _audit(session, item_id, moderator_id, action)
        session.commit()
        return "Отклонено. В MAX не отправляется."

    if action == "approve":
        if item.image_id is not None and item.image_rights_decision not in (
            "remove_image", "cleared"
        ):
            return "Сначала подтверди право на фото или исключи изображение."
        if item.image_id is None and item.image_rights_decision != "no_image":
            return "Неизвестное решение о фото. Одобрение заблокировано."
        item.status = "approved"
        item.moderator_telegram_id = moderator_id
        item.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        post.status = "approved"
        _audit(session, item_id, moderator_id, action, {
            "snapshot_sha256": item.snapshot_sha256,
            "image_rights_decision": item.image_rights_decision,
        })
        session.commit()
        return "Одобрено и сохранено. MAX остаётся выключенным до отдельного включения."

    return "Неизвестное действие."


def valid_approval(session: Session, post: Post, *, require_text_only: bool = True) -> bool:
    """Publisher-level check, independent of bot callbacks and Post.status."""
    if post.status != "approved":
        return False
    item = (
        session.query(ModerationItem)
        .filter(ModerationItem.post_id == post.id, ModerationItem.status == "approved")
        .first()
    )
    if item is None or not item.moderator_telegram_id or not item.reviewed_at:
        return False
    action = (
        session.query(ModerationAction.id)
        .filter(
            ModerationAction.moderation_item_id == item.id,
            ModerationAction.action == "approve",
            ModerationAction.moderator_telegram_id == item.moderator_telegram_id,
        )
        .first()
    )
    if action is None:
        return False
    if item.snapshot_sha256 != snapshot_hash(post.post_text, item.image_id):
        return False
    if item.image_id is None:
        return item.image_rights_decision == "no_image"
    if item.image_rights_decision == "remove_image":
        return True
    # Existing MAX sender is text-only. Never silently drop approved photos:
    # keep these items awaiting a future image-capable publisher.
    return not require_text_only and item.image_rights_decision == "cleared"
