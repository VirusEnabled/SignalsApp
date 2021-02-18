from datetime import datetime, timedelta
from.TradierDataFetcherService import  *
import statistics as stats
import plotly.graph_objects as p_go
from plotly.subplots import make_subplots
import pandas as pd

TRADIER_API_OBJ = TradierDataHandler()

def generate_candle_sticks_graph(stock_historical_data, stock_details):
    """
    generated the basic candlestick diagrams based ont
    the stock values for the ohlcv
    :param stock: dict with all values needed
    :param stock_details: dict: all details related to the stock being plotted
    :return: plotly graph
    """

    df = pd.read_json(json.dumps(stock_historical_data))
    fig_candlestick = make_subplots(specs=[[{"secondary_y": True}]])
    fig_candlestick.add_trace(p_go.Candlestick(x=df['date'],
                                 open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close']),
                  secondary_y=True)

    fig_candlestick.add_trace(p_go.Bar(x=df['date'], y=df['volume']),
                  secondary_y=False)

    fig_candlestick.update_layout(
        title=f'{stock_details["Name"]} Market',
        yaxis_title=f'{stock_details["Code"]} Stock',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig_candlestick.update_traces(name='OHLC', selector=dict(type='candlestick'))
    fig_candlestick.update_traces(name='Volume', selector=dict(type='bar'),opacity=0.40)

    fig_candlestick.layout.yaxis2.showgrid = False
    return fig_candlestick.to_html(full_html=False, default_height=700, default_width=1400)

def load_company_details(symbol):
    """
    loads company details
    :param symbol: str
    :return: dict
    """
    result = {'error':'The stock was not found, try a different one.'}
    flag = False
    for record in TRADIER_API_OBJ.load_markets():
        if record['Code'] == symbol:
            result = record
            flag = True
    return flag, result


def get_stock_historical_candlestick_graph(symbol, start_date=None,
                                           end_date=None, interval='daily'):
    """
    generates the graph needed in order to display the historical data
    based on the given market
    :param symbol: str: market
    :param start_date: date interval to parse from
    :param end_date: date interval to parse to
    :param interval: str: it could be daily, weekly or monthly
    :return: tuple
    """
    result = None
    status, historical = TRADIER_API_OBJ.get_historical_data(symbol=symbol, interval=interval,
                                                             start_date=start_date,
                                                             end_date=end_date)

    # stock data comming from there.
    st, stock_details = load_company_details(symbol=symbol)

    if status and st:
        # print(historical['history']['day'], 'here')
        # print(stock_details)
        result = generate_candle_sticks_graph(historical['history']['day'], stock_details)
    else:
        status  = False
        result = historical if not status else stock_details

    # print(status, result)
    return status, result



def verify_stock(stock):
    """
    verifies the stock
    :param stock: str: symbol
    :return: tuple
    """
    flag, result = False, {}



    return flag, result

def generate_graph_calculations(ohlcv: list) -> p_go.histogram:
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


def calculate_macd(data: list) -> pd.DataFrame:
    """
    generates the calulation for the MACD
    :param data:
    :return:
    """

def calculate_adr(data: list) -> pd.DataFrame:
    """
    generates the calulation for the ADR
    :param data:
    :return:
    """

def calculate_rsi(data: list) -> pd.DataFrame:
    """
    generates the calulation for the RSI
    :param data:
    :return:
    """


def calculate_stochastic(data: list) -> pd.DataFrame:
    """
    generates the calulation for the STOCHASTIC
    :param data:
    :return:
    """



def format_data(ohlcv: list) -> pd.DataFrame:
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
