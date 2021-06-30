from __future__ import absolute_import, unicode_literals
from celery import shared_task
from report_maker.helpers import *
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

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
        raise Exception(result['error'])

    else:
        transaction_details = calculate_transactions_for_symbols(symbols=stock_list,
                                                                 start_date=result['start_date'])
        if not transaction_details['status']:
            logger.error(f"{transaction_details['error']}")
            raise Exception(transaction_details['error'])

        else:
            logger.info(f"The task was performed successfully.")
            return f"The task was performed successfully " \
                   f"for the symbols: {stock_list}, starting at{result['start_date']}"


@shared_task
def generate_transactions_for_existing_data():
    """
    generates the transactions for the
    stocks based on the existing start_date
    :return: None
    """
    stock_list = [symbol.symbol for symbol in Stock.objects.filter(priority=Stock.choices[0]).order_by('symbol')]
    times = generate_time_intervals_for_api_query()

    if not times['status']:
        logger.error(f"{times['error']}")
        raise Exception(times['error'])
    result = calculate_transactions_for_symbols(symbols=stock_list,
                                                start_date=times['start_date'])
    if not result['status']:
        logger.error(f"{result['error']}")
        raise Exception(result['error'])

    else:
        logger.info(f"The task was performed successfully.")
        return f"The task was performed successfully " \
               f"for the symbols: {stock_list}, starting at{result['start_date']}"