from __future__ import absolute_import, unicode_literals
import sys
from pathlib import Path
from celery import shared_task
main_dir = Path(__file__).resolve().parent.parent
sys.path.append(main_dir)
from report_maker.helpers import *


@shared_task
def test_celery():
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
    stock_list = Stock.objects.all().order_by('priority')
    result = fetch_markets_data(symbols=stock_list)
    # perform action or log activity here
    if not result['status']:
        pass
    else:
        pass


