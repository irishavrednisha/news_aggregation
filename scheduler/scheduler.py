"""Collection-only Windows scheduler. ML runs separately, optionally in WSL."""
import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from main import collect

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
is_running = False


def run_news_pipeline():
    global is_running
    if is_running:
        logging.warning("Collection already running; skipping overlapping run")
        return
    is_running = True
    try:
        collect()
    except Exception:
        logging.exception("Collection failed; next scheduled run remains available")
    finally:
        is_running = False


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Europe/Moscow")
    scheduler.add_job(
        run_news_pipeline,
        trigger=IntervalTrigger(minutes=30),
        id="news_collection",
        name="Collection only (no ML or MAX)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(),
    )
    logging.info("Collection scheduler started; ML and MAX are not loaded")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logging.info("Scheduler stopped")
