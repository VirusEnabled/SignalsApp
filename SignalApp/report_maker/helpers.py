import plotly
import pandas
from datetime import datetime
from.TradierDataFetcherService import  *

TRADIER_API_OBJ = TradierDataHandler()

def generate_candle_sticks_graph(stock):
    """
    generated the basic candlestick diagrams based ont
    the stock values for the ohlcv
    :param stock: dict with all values needed
    :return: plotly graph
    """


def generate_graph_calculations(ohlcv: list) -> plotly.graph_objs.histogram:
    """
    generates a graph based on the given
    data in order to determine
    certain values which  how the values are
    behaving

    operations to make on the values:
     1.MACD
    2.ADR
    3.RSI
    4.STOCHASTIC
    :param ohlcv:list :values related to the historical data on the stock
    :return: graph historgram
    """


def format_data(ohlcv: list) -> pandas.DataFrame:
    """
    formats the data coming from the dict
    and turns it into a dataframe
    :param ohlcv: list of values
    :return: pandas DataFrame Object
    """



def schedule_auto_save(schedule: datetime) -> bool:
    """
    sends the signal to celery
    in order to schedule the auto save
    to the models needed in order
    :param schedule:
    :return:
    """



def cache_info(info: dict) -> bool:
    """
    saves the values provided in redis
    based on the given user and keeps it
    there so that we save it later.
    :param info:
    :return:
    """
