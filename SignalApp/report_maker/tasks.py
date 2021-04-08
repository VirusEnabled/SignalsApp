from __future__ import absolute_import, unicode_literals
from celery import shared_task
from report_maker.helpers import *
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@shared_task
def test_celery():
    logger.info(f"The task was performed successfully, HELLO CELERY!!!")
    return "Hello Celery!!"


@shared_task
def load_stock_data_to_db():
    """
    loads the stock data based on
    the incoming data from the API
    and performs calculations in the
    indicators therefore after that
    this execution is performed hourly
    based on the given datetime and schedule
    provided in the configurations.
    :return: None
    """
    stock_list = [symbol.symbol for symbol in Stock.objects.filter(priority=Stock.choices[0]).order_by('symbol')]
    result = fetch_markets_data(symbols=stock_list)
    if not result['status']:
        logger.error(f"{result['error']}")
        # raise Exception(result['error'])
    else:
        logger.info(f"The task was performed successfully.")


