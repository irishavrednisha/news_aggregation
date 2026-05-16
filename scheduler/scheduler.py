import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from main import main


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


is_running = False


def run_news_pipeline():
    global is_running

    if is_running:
        logging.warning("Предыдущий запуск ещё не завершён. Новый запуск пропущен.")
        return

    is_running = True

    try:
        logging.info("Запуск сбора новостей...")
        main()
        logging.info("Сбор новостей завершён.")

    except Exception as error:
        logging.exception(f"Ошибка во время выполнения пайплайна: {error}")

    finally:
        is_running = False


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Europe/Moscow")

    scheduler.add_job(
        run_news_pipeline,
        trigger=IntervalTrigger(minutes=30),
        id="news_pipeline",
        name="Сбор и обработка новостей",
        replace_existing=True,
        max_instances=1,
        next_run_time=datetime.now(),
    )

    logging.info("Планировщик запущен. Новости будут собираться каждые 30 минут.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logging.info("Планировщик остановлен вручную.")