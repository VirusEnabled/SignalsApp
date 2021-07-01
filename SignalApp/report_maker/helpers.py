import pytz
from .DataFetcherService import  *
import plotly.graph_objects as p_go
from plotly.subplots import make_subplots
import pandas as pd
import json
from .models import *
import pdb
from .serializers import *
import calendar
# from talipp.indicators import RSI, MACD, Stoch
from technical_indicators_lib import RSI, MACD,StochasticKAndD
from pandas_ta import rsi, stoch, macd,sma
import numpy as np
import requests as r
DATA_API_OBJ = APIDataHandler()
import time
import random
from datetime import timezone

class Node(object):
    def __init__(self, value):
        self.prior = None
        self.value = value
        self.next = None

    def to_list(self) -> list:
        """
        turns the node to
        a list of items
        :return: list
        """
        current = self
        values = []
        while current:
            values.append(current.value)
            current = current.next
        return values




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
        title=f'{stock_details["symbol"]} Market',
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


def get_year_weekends():
    """
    gets the current year's weekends dates
    :return: list
    """
    weekends = []
    current_year = date.today().year
    for month in range(1,13):
        for week in calendar.monthcalendar(current_year,month=month):
            if week[-2] != 0:
                weekends.append(date(year=current_year,month=month,day=week[-2]).isoformat())

            if week[-1] != 0:
                weekends.append(date(year=current_year,month=month,day=week[-1]).isoformat())
    return weekends

def generate_live_graph(stock_data:pd.DataFrame, stock_details, option: str='olhcv'):
    """
    generated the basic candlestick diagrams based ont
    the stock values for the ohlcv
    :param stock: dict with all values needed
    :param stock_details: dict: all details related to the stock being plotted
    :param option: generates a graph based on the selection itself, the options are:
    -olhcv
    -rsi
    -adr
    -stochastic
    -macd
    :return: plotly graph
    """
    # status, calendar = DATA_API_OBJ.get_market_calendar("2020")
    #
    df = stock_data
    graph =  make_subplots(specs=[[{"secondary_y": True}]])

    if option == 'olhcv':
        inner_graph = p_go.Candlestick(x=df['datetime'],
                                       open=df['open'], high=df['high'],
                                       low=df['low'], close=df['close'])
        graph.add_trace(inner_graph,
                        secondary_y=True)
        # graph.add_trace(p_go.Bar(x=df['datetime'], y=df['volume']),
        #                 secondary_y=False)
        graph.update_traces(name='OHLC', selector=dict(type='candlestick'))
        # graph.update_traces(name='Volume', selector=dict(type='bar'), opacity=0.70)
        graph.update_layout(height=750)

        graph.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[22, 7], pattern="hour"),
                dict(values=get_year_weekends()+["2020-12-25", "2021-01-01"])  # hide holidays (Christmas and New Year's, etc)
            ]
        )
        # graph.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

    elif option == 'rsi':
        graph.add_trace(p_go.Scatter(x=df['datetime'], y=df['operation_data']),
                        secondary_y=True)
        graph.update_traces(name='RSI',showlegend=True ,selector=dict(type='scatter'))
        graph.update_layout(height=300)

    elif option == 'stochastic':
        graph.add_trace(p_go.Scatter(x=df['datetime'], y=df['k_fast'],
                                     y0=df['d_slow'],
                                     name='k_fast'),
                        secondary_y=True)
        graph.update_traces(name='K_FAST',showlegend=True ,selector=dict(type='scatter',
                                                                         name='k_fast'),line_color='green')

        graph.add_trace(p_go.Scatter(x=df['datetime'], y=df['k_slow'],
                                     name='k_slow'),
                        secondary_y=True)
        graph.update_traces(name='K_SLOW', showlegend=True, selector=dict(type='scatter',
                                                                          name='k_slow'), line_color='magenta')
        graph.update_layout(height=300)


    elif option == 'macd':
        graph.add_trace(p_go.Scatter(x=df['datetime'], y=df['macd'],
                                     name='MACD'),
                        secondary_y=True)
        graph.update_traces(name='MACD', showlegend=True, selector=dict(type='scatter',
                                                                          name='MACD'),
                            line_color='red')
        graph.add_trace(p_go.Scatter(x=df['datetime'], y=df['signal'],
                                     name='Signal Line'),
                        secondary_y=True)

        graph.update_traces(name='Signal Line', showlegend=True,
                            selector=dict(type='scatter',
                                          name='Signal Line'), line_color='purple')

        graph.update_layout(height=300)

    elif option == 'adr':
        graph.add_trace(p_go.Scatter(x=df['datetime'], y=df['operation_data'],
                                     name='adr'),
                        secondary_y=True)
        graph.update_traces(name='adr', showlegend=True, selector=dict(type='scatter',
                                                                          name='adr'),
                            line_color='orange')

        graph.update_layout(height=300)


    graph.update_layout(
        title=f'{stock_details["symbol"]} Market: {option.capitalize()} graph',
        yaxis_title=f'{stock_details["symbol"]} Stock',
        legend=dict(
            yanchor="top",
            y=0.999,
            xanchor="left",
            x=-0.05
        )
    )

    graph.update_layout(xaxis_rangeslider_visible=False)

    # graph.layout.yaxis2.showgrid = True

    return graph


def update_graph(graph, new_values, option):
    """
    updates the given graph with the new values
    :param graph: plotly figure
    :param new_values: dataframe
    :param option: this is going to depend whether is : ohlcv or stochastic or rsi or macd
     1.macd
    2.adr
    3.rsi
    4.stochastic
    5.ohlcv

    :return: tuple
    """
    status, result = False, {}
    try:
        if option == "rsi":
            graph.add_trace(p_go.Scatter(x=graph.data[0]['x'],y=new_values),
                  secondary_y=False)
            #
            graph.update_traces(name=option.title(), selector=dict(type='scatter'))
            result = graph
            status = True

        elif option == 'stochastic':
            graph.add_trace(p_go.Scatter(x=new_values['datetime'], y=new_values['d_fast']),
                            secondary_y=False)
            #
            graph.update_traces(name=option.title(), selector=dict(type='scatter'))
            result = graph
            status = True
        else:
            pass

    except Exception as X:
        result['error'] = f"There has been an error with your request: {X}"

    finally:
        return status, result

def render_graph(graph):
    """
    converts the graph to a string
    so that it could be processed by django
    :param graph:
    :return:
    """
    return graph.to_html(full_html=False, default_height=700, default_width=1400)


def load_company_details(symbol):
    """
    loads company details
    :param symbol: str
    :return: dict
    """
    result = {'error':'The stock was not found, try a different one.'}
    flag = False
    for record in DATA_API_OBJ.load_markets():
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
    status, historical = DATA_API_OBJ.get_historical_data(symbol=symbol, interval=interval,
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

class RSIndicator(RSI):
    def __init__(self):
        super().__init__()

    def get_value_list(self, close_values: pd.Series, time_period: int = 14)->pd.Series:
        """
        overriding the main methods so that we can
        have this utility work because it's providing error
        :param close_values: series of the
        :param time_period: int
        :return: pd.Series
        """
        self.df["close"] = close_values
        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["GAIN"] = 0.0
        self.df["LOSS"] = 0.0

        self.df.loc[self.df["close"].astype(float) > self.df["close_prev"].astype(float),
                    "GAIN"] = self.df["close"].astype(float) - self.df["close_prev"].astype(float)

        self.df.loc[self.df["close_prev"].astype(float) > self.df["close"].astype(float),
                    "LOSS"] = self.df["close_prev"].astype(float) - self.df["close"].astype(float)

        self.df["AVG_GAIN"] = self.df["GAIN"].ewm(span=time_period).mean()
        self.df["AVG_LOSS"] = self.df["LOSS"].ewm(span=time_period).mean()

        self.df["AVG_GAIN"].iloc[:time_period] = np.nan
        self.df["AVG_LOSS"].iloc[:time_period] = np.nan

        self.df["RS"] = self.df["AVG_GAIN"] / (self.df["AVG_LOSS"] + 0.000001)  # to avoid divide by zero
        rsi_values = 100 - ((100 / (1 + self.df["RS"])))

        self.df = pd.DataFrame(None)

        return rsi_values.dropna()

class Stoch(StochasticKAndD):
    def __init__(self):
        super().__init__()

    def get_value_df(self, df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low and close values\n
            time_period(int): look back time period \n

        Returns:
            pandas.DataFrame: new pandas dataframe adding d and k as new columns,
            preserving the columns which already exists\n
        """

        df["highest high"] = df["high"].rolling(
            window=time_period).max()
        df["lowest low"] = df["low"].rolling(
            window=time_period).min()
        df["k"] = 100 * ((df["close"].astype(float) - df["lowest low"].astype(float)) /
                              (df["highest high"].astype(float) - df["lowest low"]).astype(float))
        df["d"] = df["k"].rolling(window=3).mean()

        df = df.drop(["highest high", "lowest low"], axis=1)
        return df


def get_last_records(symbol:str, first_record_date: str ,window_number:int=14):
    """
    gets the last 14 records related to the stock to make all
    calculations so that it doesn't affect the input of the data,
    this only works when refreshing the data
    :param symbol: str
    :param first_record_date: str: the date of the first record to get the past 14 items
    :param window_number: int
    :return:list of dict
    """
    result = {}
    try:
        stock = Stock.objects.get(symbol=symbol)
        historical_data = stock.historicaldata_set.filter(api_date__lt=datetime.fromisoformat(
            first_record_date)).order_by('-api_date')
        records = HistoricalDataSerializer(instance=historical_data,many=True).data[:window_number
                  if window_number<=len(historical_data) else 14]
        records.reverse()
        result['data'] = records
        result['status'] = True

    except Exception as X:
        result['status'] = False
        result['error'] = f"There was an internal Error: {X}"

    return result

# got issues with the amount of receiving amount of data.
def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    """
    generates the calulation for the MACD
    :param data:
    :return: dataframe
    """
    ema12 = data['close'].ewm(span=8, adjust=False).mean()
    ema26 = data['close'].ewm(span=21, adjust=False).mean()
    macdx = ema12 - ema26
    signal = macdx.ewm(span=5, adjust=False).mean()
    resultant_data = macd(close=data['close'].astype(float),fast=8, slow=21,signal=5)
    # ()

    result = pd.DataFrame(data={'macd': macdx, 'signal':signal,'datetime':data['datetime']})
    # result = pd.DataFrame(data={'macd': resultant_data['MACD_8_21_5'],
    #                             'signal':resultant_data['MACDs_8_21_5'],
    #                             'datetime':data['datetime']})
    return result


def calculate_adr(dt: pd.DataFrame) -> pd.Series:
    """
    generates the calulation for the ADR
    :param data:
    :return: pandas DataFrame

    """
    data = dt.copy()
    high = data['high'].astype(dtype=float)
    low = data['low'].astype(dtype=float)
    close = data['close'].astype(dtype=float)
    # data['tr0'] = abs(high - low)
    # data['tr1'] = abs(high - close.shift())
    # data['tr2'] = abs(low - close.shift())
    # tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    # # tr = data['tr0']/14
    # adr = lambda values, n: values.ewm(alpha=1/n, adjust=False).mean()
    # dt['adr'] = data['tr0']/14
    # print(dt)
    #
    # return data['tr1']/14
    # return adr(tr,14)
    window = 7
    adr = sma(high, window) - sma(low, window)
    return adr


def calculate_rsi(data: pd.DataFrame) -> pd.Series:
    """
    generates the calulation for the RSI
    :param data:
    :return: float
    """
    window_length = 14  # this should change based on the needs
    # rsi_holder= RSIndicator()
    # rsix = rsi_holder.get_value_list(close_values=data['close'],
    #                                 time_period=window_length)
    rsix = rsi(close=data['close'].astype(float),
               length=window_length)
    # delta = data['close'].astype(dtype=float).diff()
    # up = delta.clip(lower=0)
    # down = -1 * delta.clip(upper=0)
    # ema_up = up.ewm(com=window_length, adjust=False).mean()
    # ema_down = down.ewm(com=window_length, adjust=False).mean()
    # rs = ema_up / ema_down
    # rsi = 100.0 - (100.0 / (1.0 + rs))
    # # rsi = rsi.dropna()
    # rsi = rsi.dropna()
    # print(rsi)
    return rsix


def calculate_stochastic(data:pd.DataFrame ,**kwargs) -> pd.DataFrame:
    """
    generates the calulation for the STOCHASTIC
        Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100

    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic

    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal
    When the %K crosses below %D, sell signal
    :param data: dataframe
    :return: dataframe


STOCHASTIC
"meta": {
        "symbol": "AAPL",
        "interval": "1h",
        "currency": "USD",
        "exchange_timezone": "America/New_York",
        "exchange": "NASDAQ",
        "type": "Common Stock",
        "indicator": {
            "name": "STOCH - Stochastic Oscillator",
            "fast_k_period": 14,
            "slow_k_period": 3,
            "slow_d_period": 3,
            "slow_kma_type": "SMA",
            "slow_dma_type": "SMA"
        }
    },

    """
    # print(kwargs)
    #
    # endpoint = f"https://api.twelvedata.com/stoch?symbol={kwargs['symbol']}&interval=1h" \
    #            f"&apikey=eb61c42448454dc5b6b6f59dfe6d8072&source=docs&slow_k_period=3" \
    #            f"&start_date={kwargs['start_date'].date()}&fast_k_period=14&include_ohlc=true"
    # # headers = {
    # #     "symbol": symbol,
    # #     "interval": "1h",
    # #     "currency": "USD",
    # #     "exchange_timezone": "America/New_York",
    # #     "exchange": "NASDAQ",
    # #     "type": "Common Stock",
    # #     'start_date':kwargs['start_date'],
    # #     "indicator": json.dumps({
    # #         "name": "STOCH - Stochastic Oscillator",
    # #         "fast_k_period": 14,
    # #         "slow_k_period": 3,
    # #         "slow_d_period": 3,
    # #         "slow_kma_type": "SMA",
    # #         "slow_dma_type": "SMA"
    # #     })
    # # }
    # response = r.get(url=endpoint)
    # response_data = response.json()
    # if response_data['status'] == 'ok':
    #     response_values = response_data['values']
    #     response_values.reverse()
    #     if stored:
    #         #
    #         response_values = kwargs['full_data']['model_data'] + response_values
    #     stochastic = pd.DataFrame([
    #                                   {
    #                                     'k': d['slow_k' if 'slow_k' in d.keys() else 'k'],
    #                                     'datetime': d['datetime']
    #                                    } for d in response_values if float(d['volume']) > 0.00])
    #     stochastic['k'] = stochastic['k'].astype(float)
    # else:
    #     raise Exception(response_data['message'])
    #     # kwargs['start_date'] = kwargs['start_date'] - timedelta(days=1)
    #     # return calculate_stochastic(data=data,
    #     #                             stored=stored,
    #     #                             symbol=kwargs['symbol'],
    #     #                             start_date=kwargs['start_date'],
    #     #                             model_data=kwargs['model_data'] if stored else None,)
    time.sleep(15)
    stored = kwargs['stored']
    stochastic = pd.DataFrame([
                                  {
                                      'k': d['k'],
                                      'datetime': d['datetime']
                                  } for d in kwargs['full_data']
                                  # if float(d['volume']) > 0.00  # commented as needed to solve issue

                                  ])
    stochastic['k'] = stochastic['k'].astype(float)
    # print(stochastic)
    #

    # k = 14 # days of the window
    # d = 3 # 3 days of SMA which come from K
    # indicator = Stoch()
    # stochastic = indicator.get_value_df(df=data,
    #                                     time_period=k)
    # low_min = data['close'].rolling(window=k).min().dropna()
    # high_max = data['close'].rolling(window=k).max().dropna()
    #
    #
    # stochastic = stoch(high=high_max.astype(float),
    #                          low=low_min.astype(float),
    #                          close=data['close'].astype(float),
    #                          k=k,
    #                          d=d
    #                    )
    #
    # stochastic['datetime'] = data['datetime']
    # stochastic['k'] = stochastic['STOCHk_14_3_3']
    # stochastic['k'] = stochastic['STOCHd_14_3_3']
    #


    # stochastic = data.copy().dropna()
    # low_min = stochastic['close'].rolling(window=k).min().dropna()
    # high_max = stochastic['close'].rolling(window=k).max().dropna()
    #
    # # K value of stochastic.
    # stochastic['k'] = 100 * ((stochastic['close'].astype(dtype=float) - low_min.astype(dtype=float)) /
    #                    (high_max.astype(dtype=float) -low_min.astype(dtype=float)))
    # stochastic['d'] = stochastic['k'].rolling(window=d).mean()

    # stochastic['k_fast'] = 100 * (stochastic['close'].astype(dtype=float) -
    #                               low_min.astype(dtype=float)) / (high_max.astype(dtype=float) -
    #                                                               low_min.astype(dtype=float))
    # stochastic['d_fast'] = stochastic['k_fast'].rolling(window=d).mean()
    #
    # # Slow Stochastic
    # stochastic['k_slow'] = stochastic["d_fast"]
    # stochastic['d_slow'] = stochastic['k_slow'].rolling(window=d).mean()
    # return final_stochastic
    #

    return stochastic

# bug here
def generate_statistical_indicators(data: dict, stored:bool =False,**kwargs) -> dict:
    """
    generates the 4 statistical
    indicators for the data incoming for the available
    stocks in the market
    :param data: dict containg all data related to the calculation
    :param stored: flag to determine to either sto
    :return: dict
    """
    result = {}
    try:
        processed_data = {}
        final_container = []
        if not stored:
            processed_data = pd.DataFrame(data=data['api_data'])

        else:
            #
            final_container = data['model_data'] + data['api_data']
            processed_data = pd.DataFrame(data=final_container).dropna()

        # ()

        result = dict(stochastic=calculate_stochastic(data=processed_data,
                                                      stored=stored,
                                                      symbol=kwargs['symbol'],
                                                      start_date=kwargs['start_date'],
                                                      full_data=final_container if stored else data['api_data'],
                                                      ),
                      rsi=to_data_frame(operation_data=calculate_rsi(processed_data),
                                        time_data=processed_data['datetime']),
                      macd=calculate_macd(processed_data),
                      adr=to_data_frame(operation_data=calculate_adr(processed_data),
                                        time_data=processed_data['datetime'])
                      )
        result['olhcv'] = processed_data

        # if stored:
        #     result['rsi'] = result['rsi'].dropna()
        #     result['stochastic'] = result['stochastic'].dropna()
        #     result['adr'] = result['adr'].dropna()
        #     result['macd'] = result['macd'].dropna()
        # else:
        result['rsi'] = result['rsi'].fillna(0.00)
        result['stochastic'] = result['stochastic'].fillna(0.00)
        result['adr'] = result['adr'].fillna(0.00)
        result['macd'] = result['macd'].fillna(0.00)
        result['olhcv']['datetime'] = clean_stock_datetime(result['olhcv']['datetime'])
        result['rsi']['datetime'] = clean_stock_datetime(result['rsi']['datetime'])
        result['stochastic']['datetime'] = clean_stock_datetime(result['stochastic']['datetime'])
        result['adr']['datetime'] = clean_stock_datetime(result['adr']['datetime'])
        result['macd']['datetime'] = clean_stock_datetime(result['macd']['datetime'])

        result['status'] = True
        #

    except Exception as X:
        # raise Exception(X)
        #
        result['error'] = f"There was an error in the execution: {X}"
        result['status'] = False
        result['traceback'] = X.__traceback__
        # ()

    return result

def to_data_frame(operation_data, time_data):
    """
    creates a dataframe to plot the results of the calculations
    :param operation_data: data used to calculate the indicators
    :param time_data: datetime elapsed
    :return: dataframe object
    """
    dataframe = pd.DataFrame()
    dataframe['datetime'] = time_data
    dataframe['operation_data'] = operation_data
    return dataframe

def store_data(stock_details:dict,
               operation_data:pd.DataFrame,
               operation:str) -> tuple:
    """
    saves the data in the models for further processing
    :param stock_details: details about the stock to generate the graph for
    :param operation_data: data frame to be exported to json
    :param operation: type of operation, the available options are:
    -adr
    -stochastic
    -rsi
    -macd
    :return:tuple
    """
    status, error = False, ""
    try:
        stock = Stock.objects.get(symbol=stock_details['symbol']) if Stock.exists(stock_details['symbol'])\
            else Stock.objects.create(symbol=stock_details['symbol'],
                                      stock_details=json.dumps(stock_details))

        stock.indicatorcalculationdata_set.create(indicator=operation.capitalize(),
                                                  operation_data=operation_data.to_json())

        stock.historicaldata_set.create()
        status = True
    except Exception as X:
        error = f"There was an error with the request: {X}"
        print("ERROR HERE", X)

    finally:
        return status, error

def format_float_value(value):
    """
    removes the tuples
    :param value:
    :return: float
    """
    return round(value[0],4) if isinstance(value,tuple) else round(value,4)

def evaluate_integrity(olhcv,rsi,adr,macd,stochastic):
    """
    validates that the given data in the given date
    is concurrent in it
    :param olhcv:
    :param rsi:
    :param adr:
    :param macd:
    :param stochastic:
    :return: bool
    """
    return olhcv['datetime'] == macd['datetime']==rsi['datetime']==adr['datetime']==stochastic['datetime']

def format_date(datetime_str):
    """
    turns the datetime_str to isoformat str
    :param datetime_str: str
    :return: str
    """
    result = ""
    if '/' in datetime_str:
        dx = datetime_str.split(' ')
        datex, hour = dx[0].split('/'), dx[1].split(':')
        result= datetime(year=int(f"{'20'+datex[-1] if len(datex[-1]) < 4 else datex[-1]}"),
                        month=int(datex[1]),day=int(datex[0]),
                        hour=int(hour[0])-1, minute=int(hour[1])).isoformat()
    else:
        stamp = datetime.fromisoformat(datetime_str)
        result = datetime(year=stamp.year,
                 month=stamp.month, day=stamp.day,
                 hour=stamp.hour-1, minute=stamp.minute).isoformat()
    return result

def format_dt(dataframe: pd.DataFrame):
    """
    generates a new dataframe based on the given
    key. this is segregate the data based on the given symbol
    :param dataframe: pandas dataframe
    :param key: str: the symbol to query
    :return: dataframe
    """
    resultant = dataframe.to_dict()
    # print(resultant)

    for key in dataframe['underlying_symbol'].index:
        if (resultant['open'][key] <= 0 and
            resultant['high'][key] <= 0 and
            resultant['low'][key] <= 0 and
            resultant['close'][key] <= 0):
            for category in ['open','close','low','high',
                             'quote_datetime','vwap','ask',
                             'bid','trade_volume','underlying_symbol']:
                del resultant[category][key]


    for key, value in resultant['quote_datetime'].items():
        resultant['quote_datetime'][key] = format_date(value)
    resultant['datetime'] = resultant['quote_datetime']
    resultant['volume'] =  resultant['trade_volume']
    del resultant['quote_datetime']
    del resultant['trade_volume']

    return pd.DataFrame(resultant)


# issue here with the formating seems to be done because of strftime.
def clean_stock_datetime(date_list: pd.Series) -> pd.Series:
    """
    cleans the datetime field by just adding
    30 mins to each date time that doesn't have 30 mins.
    in it.
    :param date_list: Series
    :return:Series
    """
    for i in range(len(date_list)):
        date_eval = datetime.fromisoformat(date_list[i])
        if date_eval.minute == 30:
            break
        serialized_date = datetime(year=date_eval.year,
                                month=date_eval.month,
                                day=date_eval.day,
                                hour=date_eval.hour,
                                minute=30,
                                second=date_eval.second).isoformat()
        date_list[i] = serialized_date.replace("T"," ")

        # date_list[i] = datetime(year=date_eval.year,
        #                       month=date_eval.month,
        #                       day=date_eval.day,
        #                       hour=date_eval.hour,
        #                       minute=30,
        #                       second=date_eval.second).strftime("%Y-%m-%d %H:%m:%S")
    #
    return date_list

def process_file_data(csvfile, delimiter=";"):
    """
    saves the csv file data, formats it
    and stores it in the db.
    :param csvfile:  path of the csv file
    :return: bool
    """
    try:
        file = open(csvfile, 'r')
        dataframe = pd.read_csv(file, delimiter=delimiter).dropna()
        companies = set(dataframe['underlying_symbol'])
        groups = dataframe.groupby('underlying_symbol')
        dataframes = [
            {'data':format_dt(groups.get_group(symbol)),
             'stock_details': {'Code': symbol}
             }
            for symbol in companies
        ]
        flag, result = False, None
        for dframe in dataframes:
            proccesed_data = dframe['data']
            operations = dict(stochastic=calculate_stochastic(data=proccesed_data),
                              rsi=to_data_frame(operation_data=calculate_rsi(proccesed_data),
                                                time_data=proccesed_data['datetime']),
                              macd=calculate_macd(proccesed_data),
                              adr=to_data_frame(operation_data=calculate_adr(proccesed_data),
                                                time_data=proccesed_data['datetime']).dropna())
            operations['olhcv'] = proccesed_data
            operations['rsi'] = operations['rsi'].fillna(0)
            operations['stochastic'] = operations['stochastic'].fillna(0)
            print([(k, len(obj)) for k, obj in operations.items()])
            #
            status, error = store_full_data(dframe['stock_details'], operations)
            if not status:
                result = error
                break
        else:
            flag = True
            result = 'Success'
    except Exception as EX:
        flag= False
        result = f"ERROR: {EX}"
    return flag, result


def store_full_data(stock_details:dict,
               operation_data:dict, update:bool=False) -> tuple:
    """
    saves the data in the models for further processing
    :param stock_details: details about the stock to generate the graph for
    :param operation_data: data frame to be exported to json
    :param operation: type of operation, the available options are:
    -adr
    -stochastic
    -rsi
    -macd
    :param: update: bool: validates that the values we're passing already exist therefore, we
    shouldn't modify those who already exist
    :return:tuple
    """
    status, error = False, ""
    try:
        stock, created = Stock.objects.get_or_create(symbol__exact=stock_details['symbol'])
        if created:
            stock.stock_detail = json.dumps(stock_details)
            stock.save()

        print(len(operation_data['olhcv']),
              len(operation_data['rsi']),
              len(operation_data['macd']),
              len(operation_data['adr']),
              len(operation_data['stochastic']))

        for i in range(len(operation_data['olhcv'])):
            if evaluate_integrity(operation_data['olhcv'].iloc[i],operation_data['rsi'].iloc[i],
                                  operation_data['adr'].iloc[i],operation_data['macd'].iloc[i],
                                  operation_data['stochastic'].iloc[i]):

                """
                compra = Si RSI > 50 and k > 20 and macd > signal and rsi_actual > rsi_anterior
                Venta = Si RSI < 50 and k < 80 and macd < signal and rsi_actual < rsi_anterior
                """
                f_stoch = ''

                if (operation_data['rsi'].iloc[i]['operation_data'] > 50.00 and
                            operation_data['stochastic'].iloc[i]['k'] > 20.00 and
                            operation_data['macd'].iloc[i]['macd'] > operation_data['macd'].iloc[i]['signal'] and
                            operation_data['rsi'].iloc[i]['operation_data'] >
                            operation_data['rsi'].iloc[i - 1]['operation_data']):
                    f_stoch = 'COMPRA'

                if (operation_data['rsi'].iloc[i]['operation_data'] < 50.00 and
                            operation_data['stochastic'].iloc[i]['k'] < 80.00 and
                            operation_data['macd'].iloc[i]['macd'] < operation_data['macd'].iloc[i]['signal'] and
                            operation_data['rsi'].iloc[i]['operation_data'] <
                            operation_data['rsi'].iloc[i - 1]['operation_data']):
                    f_stoch = 'VENTA'

                # f_stoch = 'COMPRA' if operation_data['stochastic'].iloc[i]['k'] > 20.00 else 'VENTA'
                f_rsi = 'COMPRA' if operation_data['rsi'].iloc[i]['operation_data'] > 50.00 else 'VENTA'
                f_macd = 'COMPRA' if operation_data['macd'].iloc[i]['macd'] > operation_data['macd'].iloc[i]['signal'] \
                    else 'VENTA'


                if not f_stoch:
                    if f_rsi == f_macd == 'COMPRA' :
                        f_stoch = "COMPRA"

                    elif f_rsi == f_macd == 'VENTA':
                        f_stoch = "VENTA"

                    else:
                        f_stoch = random.choice([f_rsi, f_macd])

                bullet = 'ROJO' if f_stoch == f_rsi == f_macd == 'VENTA' else 'AZUL' \
                    if f_stoch == f_rsi == f_macd == 'COMPRA' else 'BLANCO'




                record, created = HistoricalData.objects.get_or_create(
                                                stock = stock,
                                                api_date=operation_data['olhcv'].iloc[i]['datetime']
                                                )

                rsi = float(operation_data['rsi'].iloc[i]['operation_data'])
                signal = float(operation_data['macd'].iloc[i]['signal'])
                macd = float(operation_data['macd'].iloc[i]['macd'])
                adr = float(operation_data['adr'].iloc[i]['operation_data'])
                k = float(operation_data['stochastic'].iloc[i]['k'])
                record.open = float(operation_data['olhcv'].iloc[i]['open'])
                record.high = float(operation_data['olhcv'].iloc[i]['high'])
                record.low = float(operation_data['olhcv'].iloc[i]['low'])
                record.close = float(operation_data['olhcv'].iloc[i]['close'])
                record.volume = float(operation_data['olhcv'].iloc[i]['volume'])

                if not created and update:

                    if (rsi > 0.00 and adr >0.00 and macd > 0.00 and signal > 0.00 and k >0.00 ) or (rsi > 0.00 and adr > 0.00 and macd > 0.00 and signal > 0.00):
                        #
                        record.rsi = rsi
                        record.adr = adr
                        # record.k_slow = float(operation_data['stochastic'].iloc[i]['k_slow'])
                        # record.k_fast = float(operation_data['stochastic'].iloc[i]['k_fast'])
                        record.k = k
                        record.macd = macd
                        record.signal = signal
                        record.f_stoch = f_stoch
                        record.f_rsi = f_rsi
                        record.f_macd = f_macd
                        record.bullet = bullet
                    else:
                        #
                        # print(record.rsi, record.macd,record.signal, record.k, record.adr)
                        pass
                    # print(i)
                    record.updated_at = datetime.now()
                else:
                    #
                    record.rsi = rsi
                    record.adr = adr
                    # record.k_slow = float(operation_data['stochastic'].iloc[i]['k_slow'])
                    # record.k_fast = float(operation_data['stochastic'].iloc[i]['k_fast'])
                    record.k = k
                    record.macd = macd
                    record.signal = signal
                    record.f_stoch = f_stoch
                    record.f_rsi = f_rsi
                    record.f_macd = f_macd
                    record.bullet = bullet

                record.save()

        status = True
    except Exception as X:
        error = f"There was an error with the store request: {X}"

    finally:
        return status, error


def process_data(data):
    """
    turns the 15m interval into 1h interval
    :param data:
    :return: series
    """
    x = data.to_list()
    serialized = []
    last = 0
    for i in range(4,len(x),4):
        serialized.append(sum(float(y) for y in x[last:i])/len(x[last:i]))
        # print(last,i,x[last:i],sum(x[last:i]))
        last += 4
        #
        # datetime.sleep(20)
    return pd.Series(data={k:v for k, v in enumerate(serialized)})

# this method ain't in use
def serialize_time(dtime):
    """
    serializes the datetime, basically extracts the hours
    :param dtime: Series
    :return:Series
    """
    dates = dtime.to_list()
    serialized = []
    for di in dates:
        d = datetime.fromisoformat(di)
        if d.minute == 0:
            serialized.append(d.isoformat())

    return pd.Series(data={k: v for k, v in enumerate(serialized)})

# this method ain't in use
def serialize_data(data: pd.DataFrame)->pd.DataFrame:
    """
    transforms the daily data into hourly data
    :param data: pd.data frame
    :return: data frame
    """
    final = pd.DataFrame(data={'open': process_data(data['open']), 'high': process_data(data['high']),
                           'low': process_data(data['low']), 'close': process_data(data['close']),
                           'volume': process_data(data['volume']),
                           'datetime': serialize_time(data['datetime'])}).dropna()
    return final

# this method ain't in use
def generate_graph_calculations(symbol, start_date=None,
                                           end_date=None, interval='15min') -> tuple:
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

    :param symbol: str: market
    :param start_date: date interval to parse from
    :param end_date: date interval to parse to
    :param interval : str : Interval of datetime per timesale. One of: tick, 1min, 5min, 15min
    :return: tuple
    """
    status, result = False, {'error':""}
    try:
        status, api_data = DATA_API_OBJ.live_market_data_getter(symbol=symbol, start_date=start_date,
                                                                end_date=end_date, interval=interval)

        if status:
            stock_details = api_data['stock_details']
            operations = generate_statistical_indicators(data=api_data['data'], stored=False)
            if not operations['status']:
                raise Exception(operations['error'])

            status, error = store_full_data(stock_details, operations)
            if not status:
                return status, error

            main_graph = generate_live_graph(operations['olhcv'], stock_details)
            rsi_graph = generate_live_graph(stock_data=operations['rsi'],stock_details=stock_details, option="rsi")
            stochastic_graph = generate_live_graph(stock_data=operations['stochastic'],
                                                   stock_details=stock_details, option="stochastic")
            macd_graph = generate_live_graph(stock_data=operations['macd'],
                                             stock_details=stock_details,
                                       option='macd')
            adr_graph = generate_live_graph(stock_data=operations['adr'],stock_details=stock_details,option='adr')

            result = \
            {
                'graph': render_graph(main_graph),
                'rsi_graph': render_graph(rsi_graph),
                'stochastic_graph': render_graph(stochastic_graph),
                'macd_graph': render_graph(macd_graph),
                'adr_graph': render_graph(adr_graph),
            }

        else:
            status = False
            result = api_data

    except Exception as X:
        result['error'] = f"There has been an error with your request: {X}"
        status = False
        print(status, result, "EXITING IT")

    finally:
        return status, result

def get_current_ny_time(date_time: datetime=datetime.now()):
    """
    gets the current new york time
    this works for time needed to fetch the data
    :return: str
    """
    eastern_time = pytz.timezone('US/Eastern')
    final_time = eastern_time.localize(date_time)
    final_time = final_time.astimezone(eastern_time)
    reducible = int(''.join(z for z in final_time.strftime("%z").split('0') if z!='' and z!='-'))
    return (final_time.astimezone(eastern_time)- timedelta(hours=reducible)).strftime("%Y-%m-%d %H:%M")


def generate_time_intervals_for_api_query():
    """
    generates the start_date and end_date
    for the API requests executed with celery

    the first part needed is to determine:
    1- need to get the last time the query was executed
    if there's any then we use that as a start date, else:
    we need to query 1 year from today. and the end_date will
    be the datetime snapshot of the day during the execution (datetime.now())

    the snapshot will be saved on redis and then once called again, we'll use
    the snapshot as a start_date and the end_date will be start_date + 1 hour

    # right now missing these evaluations:
        if the snapshot ends in a 19:30 then the end_date should be 1+ day
         and starts at 8:30

         if the snapshot day ends in friday then the end_date should start on monday at
         8:30pm

         if the snapshot it's on the limit time: current date-19:30, and the current date is the same
         as the snapshot, the end_date is the same snapshot.

     so basically the snapshot works closely with the current date the server has or the
     current datetime there is in new york
    :return: dict
    """
    final_result = {}
    try:
        today = get_current_ny_time(date_time=datetime.today())
        _start_date = datetime(year=datetime.now().year-1,month=1,day=4,hour=9,minute=30)
        start_date = get_current_ny_time(_start_date)
        end_date = today
        status, result = settings.REDIS_OBJ.get_last_fetched_time()
        if not status and 'error' in result.keys():
            raise Exception(result['error'])

        elif not status:
            if not result['last_refresh_time_celery']:
                final_result['start_date'] = datetime.fromisoformat(start_date)
                final_result['end_date'] = datetime.fromisoformat(end_date)
        else:
            final_result['start_date'] = datetime.fromisoformat(result['last_refresh_time_celery'])
            final_result['end_date'] = datetime.fromisoformat(end_date)
            if final_result['start_date'] > final_result['end_date']:
                s = final_result.pop('start_date')
                f = final_result.pop('end_date')
                final_result['start_date'] = f
                final_result['end_date'] = s

                # if final_result['start_date'].minute > 30:
                #     print(today)
                #
                # # update to control issue with the dates and stuffs
                if final_result['start_date'].date() == datetime.today().date() \
                        and (final_result['start_date'].hour > 18 or final_result['start_date'].hour < 9):
                    # print(today)
                    #
                    final_result['prior_start_date'] = final_result['start_date']
                    final_result['start_date'] -= timedelta(days=1)

        status, error = settings.REDIS_OBJ.refresh_last_fetched_time(new_refresh_time=datetime.fromisoformat(
            end_date).isoformat())

        if not status:
            raise Exception(error)

        final_result['status'] = status

    except Exception as E:
        final_result['error'] = f"There was an Error: {E}"
        final_result['status'] = False

    return final_result

def recalculate_stock_indicators(symbol:str) -> dict:
    """
    updates all values on the existing symbol on the db
    :param symbol: str
    :return: dict
    """
    result = {}
    try:
        stock = Stock.objects.get(symbol=symbol)
        operations = generate_statistical_indicators(data={'model_data':stock.historicaldata_set.all(),
                                                           'api_data':[]}, stored=True)
        if not operations:
            raise Exception(operations['error'])
        status, error = store_full_data(stock_details={'stock_details':stock.stock_details,
                                                       'symbol':symbol},
                                        operation_data=operations)
        if not status:
            raise Exception(error)
        result['status'] = True

    except Exception as X:
        result['error'] = f"{X}"
        result['status'] = False

    return result

def get_entry_price(index:int, repeated:int,
                    values:list) -> float:
    """
    calculates the entry price
    needed for the given value

    :param index: int
    :param repeated: int
    :param values: list
    :return: float
    """
    # entry_price = float(sum(values[j].open for j in range(len(values)) if j <= index and
    #                     j >= abs(index-repeated)) / divisor) if repeated > 1 else values[index].open
    entry_price = 0.00
    if index == 0:
        entry_price = values[0].open
    else:
        entry_price += sum(x.open for x in values[:index])/len(values[:index])

    # ()
    return entry_price

# modified for upgrade
def _get_entry_price(index:int, values:list,
                     transaction_id: int,
                     stock:object,
                     stored:bool) -> float:
    """
    calculates the entry price
    needed for the given value

    aggregating the ID of the transaction we
    just need to look up for the value,
    it's the same method, now we just look for the
    item throu the database and calculate
    based on the given ID the Avg amount which the
    sum of all round(opens / len(opens), 2)

    :param index: int
    :param repeated: int
    :param values: list
    :return: float
    """
    entry_price = 0.00
    xrange =  len(values) - 1 if len(values) > 1 else len(values)
    if index == 0 and stored:
        last_transaction = HistoricalTransactionDetail.get_last_transaction(symbol=stock.symbol)
        if last_transaction and last_transaction.status == 'open':
            opens = HistoricalTransactionDetail.get_open_values_from_open_transaction(transaction_id=transaction_id,
                                                                                  symbol=stock.symbol)
            if len(opens) == 0:
                entry_price = 0.00
            else:
                entry_price = round(sum(opens) / len(opens), 2)
        else:
            entry_price = round(values[index].open, 2)

    elif index == 0:
        entry_price = round(values[index].open, 2)
    else:
        opens = [val.open for val in reversed(values[:index+1 if index < xrange else xrange])]
        if len(opens) == 0:
            entry_price = 0.00
        else:
            entry_price = round(sum(opens) / len(opens), 2)

    return entry_price

def get_transaction_detail_status(record:HistoricalData,
                                  stop_loss:float,
                                  take_profit:float ) -> str:

    """
    Analyzes which value is required to get
    the correct transaction status based on :

     if in SALE== RED:
        stop_loss <= low: close else open

    if in BUY == BLUE:
        take_profit >= high: close else open

    :param record: Django's object
    :return: str
    """
    # print("LOW",record.low, stop_loss, stop_loss<=record.low ,"ID",record.id)
    result = 'close' if record.low <= stop_loss  or record.high >= take_profit  else 'open'
    # print("HIGH",record.high, take_profit, take_profit >= record.high  ,"ID",record.id)

    return result

def organize_transaction_data(values: list)->list:
    """
    organizes the values of the list
    based on the api_date
    :param values: list
    :return: list
    """
    records = []
    x_values = [x.api_date for x in values]
    while values:
        v = values.pop(x_values.index(min(x_values)))
        x_values.pop(x_values.index(min(x_values)))
        records.append(v)
    return records


def stack_equal(queryset):
    """
    keeps only those who are consecutively with one
    value
    :param queryset:
    :return: list of lists
    """
    records = []
    node = None
    current = None
    total = len(queryset)

    for i in range(total-1):
        if queryset[i].bullet == queryset[i+1].bullet and not node and  queryset[i+1].bullet != 'BLANCO':
            node = Node(queryset[i])
            node.next = Node(queryset[i+1])
            current = node.next
            print('starting',current.value, node.value)

        elif current and queryset[i].bullet == current.value.bullet:
            current.next = Node(queryset[i])
            current = current.next
            print('here, adding bullet', current.value)

        elif queryset[i].bullet != queryset[i+1].bullet and not node:
            print('NOTHING MATCHED')
            continue

        else:
            if node:
                values = node.to_list()
                values.pop(0)  # dropping the first ones so that the values are calculated properly.
                values = organize_transaction_data(values)
                records.append(values)
                print("SUPPOSEDLY ADDED VALUES TO LIST")
            # ()
            node = None
            current = None

    # ()
    return records

# # old version before update
def __calculate_tp_sl_on_records(start_date:datetime, symbol: str) -> dict:
    """
    calculates the take profit values and stop loss on the records
    that matches the given description
    :param start_date: datetime: the date to start fetching data and calculating from
    :return: dict
    """
    result = {'created_records':[]}
    p_i = lambda i : i - 1 if i > 0 else 0
    existing_data = Stock.objects.get(symbol=symbol).historicaldata_set.filter(api_date__gte=start_date.date())
    serialized = stack_equal(existing_data)
    last_transaction = {}

    # calculate Stop Loss(SL) and Take Profit (TP)
    repeated = 0
    try:
        for item in serialized:
            len_ = len(item)
            for i in range(len_):
                if (item[p_i(i)].bullet == item[i].bullet
                    and item[i].bullet == 'ROJO' or item[i].bullet == 'AZUL'):
                    repeated += 1
                    entry_price = get_entry_price(index=i, repeated=repeated, values=item)

                    take_profit = entry_price + (item[i].adr * 2) if item[i].bullet == "AZUL" else \
                        entry_price - (item[i].adr * 1.5)

                    stop_loss = entry_price - (item[i].adr * 1.5) if item[i].bullet == "AZUL" else \
                        entry_price + (existing_data[i].adr * 1.5)

                    record, created = HistoricalTransactionDetail.objects.get_or_create(
                        historical_data=item[i]
                    )
                    """
                          - Precio de entrada: Es el precio de open de la vela con el que inicia la transacción
                          (Solo se guardara 1 vez en la transacción con la que inicia el bloque)
                          - Precio de Salida: Este se calcula cuando se cierra la transacción y
                           se calcula asi: Si entry_type es Compra = Precio Promedio + (ADR * 2.0).
                           Si entry_type = Venta = Precio Promedio - (ADR * 1.5).
                          - Valor_Ganado/Perdido: (Precio de Salida * # de Unidades) - (Precio de entrada * # de Unidades)
                    """

                    record.stop_loss_price = stop_loss
                    record.take_profit_price = take_profit
                    record.avg_price = entry_price

                    record.status = get_transaction_detail_status(record=item[i],
                                                           stop_loss=stop_loss,
                                                           take_profit=take_profit)

                    record.entry_type = 'VENTA' if item[i].bullet == 'ROJO' else 'COMPRA'


                    # this should be added in the update version of the method
                    if record.status == 'close':
                        record.closing_price = record.avg_price + existing_data[i].adr * 2.0 if record.entry_type == 'COMPRA'\
                            else record.avg_price - existing_data[i].adr  * 1.5
                    # else:
                    #     earning_losing_value =

                    if not created:
                        record.updated_at = datetime.now()

                    record.save()

                    result['created_records'].append(record)
                else:
                    repeated = 0

        # ()
        result['status'] = True

    except Exception as X:
        result['status'] = False
        result['traceback'] = X.__traceback__
        result['err_obj'] = X
        result['error'] = f"{X}"
        # ()


    return result

def find_concurrent_patterns(dataset: list, start_index: int=0,)-> dict:
    """
    finds the pattern and returns the index where it should start
    calculating
    :param dataset:
    :param start_index:
    :return: dict

        # identify the pattern:
        # a white followed by at least 2 of either color if there's a traceback of a closed item
        # so basically we need to be able to tell whether the pattern should be applied or not
        # for that I need to know if I'm on a existing transaction or the last trasaction was closed.
        # then if I'm on a open transaction, all I need to do is to calculate then check if the sstatus still open,
        # if it changes, then I close the transaction and execute the conditioning where I'm ignoring records until
        # I meet the pattern.
    """

    result = {}
    index_list = [i for i in range(start_index, len(dataset) - 2)]
    # index_list = [dataset.index(i) for i in dataset if dataset.index(i) >= start_index]

    try:
        for i in index_list:
            # print(dataset[i].bullet,dataset[i + 1].bullet,dataset[i + 2].bullet )
            if dataset[i].bullet == 'BLANCO' and (
                            dataset[i + 1].bullet == dataset[i + 2].bullet and dataset[i+2].bullet != 'BLANCO' ):
                # execute code to gather and calculate the values
                # print(dataset[i].bullet, dataset[i + 1].bullet, dataset[i + 2].bullet,"FOUNDDDDD")
                # print(dataset[i], dataset[i + 1], dataset[i + 2])
                result['start_index'] = i + 2
                break
            # print(dataset[i].bullet,dataset[i + 1].bullet,dataset[i + 2].bullet, 'NOT FOUNDDD')

        if 'start_index' not in result.keys():
            result['status'] = False
            result['message'] = 'No items found in the pattern'
            # ()
            # raise Exception('No items found in the pattern')
        else:
            result['status'] = True

    except Exception as X:
        result['status'] = False
        result['traceback'] = X.__traceback__
        result['err_obj'] = X
        result['error'] = f"{X}"

    return result

def _calculate_tp_sl(dataset:list, stored:bool,
                     start_index: int=0, transaction_id:int=1) -> dict:
    """
    calculates the take profit and stop loss
    based on the new updates provided list of values
    with the given index

    :param dataset:
    :param start_index:
    :return: dict

    # this is related to another Model so don't concert yourself with it atm.
    - Precio de entrada: Es el precio de open de la vela con el que inicia la transacción
    (Solo se guardara 1 vez en la transacción con la que inicia el bloque)
    - Precio de Salida: Este se calcula cuando se cierra la transacción y
    se calcula asi: Si entry_type es Compra = Precio Promedio + (ADR * 2.0).
    Si entry_type = Venta = Precio Promedio - (ADR * 1.5).
    - Valor_Ganado/Perdido: (Precio de Salida * # de Unidades) - (Precio de entrada * # de Unidades)
    """
    result = {}
    first_entry_price = dataset[start_index:][0].open
    spectrum = len(dataset[start_index:])
    try:
        for idx, record in enumerate(dataset[start_index:]):
            last_transaction = HistoricalTransactionDetail.get_last_transaction(symbol=record.stock.symbol)
            entry_price = _get_entry_price(index=idx,
                                           values=dataset[start_index:],
                                           transaction_id=transaction_id,
                                           stock=record.stock,
                                           stored=stored)

            transaction, created = HistoricalTransactionDetail.objects.get_or_create(
                historical_data=record
            )
            """
                  - Precio de entrada: Es el precio de open de la vela con el que inicia la transacción
                  (Solo se guardara 1 vez en la transacción con la que inicia el bloque)
                  - Precio de Salida: Este se calcula cuando se cierra la transacción y
                   se calcula asi: Si entry_type es Compra = Precio Promedio + (ADR * 2.0).
                   Si entry_type = Venta = Precio Promedio - (ADR * 1.5).
                  - Valor_Ganado/Perdido: (Precio de Salida * # de Unidades) - (Precio de entrada * # de Unidades)
            """

            if not last_transaction:
                transaction.entry_type = 'VENTA' if record.bullet == 'ROJO' else 'COMPRA'



            elif last_transaction and last_transaction.status == 'close':
                transaction.entry_type = 'VENTA' if record.bullet == 'ROJO' else 'COMPRA'
                # print(transaction_id, last_transaction, last_transaction.transaction_id,
                #       last_transaction.entry_price,entry_price)
            else:
                transaction.entry_type = last_transaction.entry_type
            print(transaction, transaction_id, transaction.entry_type, record.bullet,
                  last_transaction.entry_price if last_transaction else "THE LAST TRANSACTION IS NONE",
                  entry_price)

            take_profit = entry_price + (float(record.adr) * 2) if transaction.entry_type == 'COMPRA' else \
                entry_price - (float(record.adr) * 1.5)

            stop_loss = entry_price - (float(record.adr) * 1.5) if transaction.entry_type == 'COMPRA' else \
                entry_price + (float(record.adr) * 1.5)


            transaction.stop_loss_price = round(stop_loss,2)
            transaction.take_profit_price = round(take_profit,2)
            transaction.avg_price = round(entry_price,2)
            transaction.transaction_id = transaction_id
            transaction.id_market = record.stock.id
            transaction.status = get_transaction_detail_status(record=record,
                                                          stop_loss=stop_loss,
                                                          take_profit=take_profit)


            # if record.api_date.isoformat() >= "2021-01-04T15:30:00":
            #     ()

            if idx == 0 and transaction.status =='open':
                transaction.entry_price = record.open
                transaction.number_of_unities = int(5000/record.open) if record.open > 0.00 else 0

            # this should be added in the update version of the method
            if transaction.status == 'close':
                transaction.closing_price = round(float(transaction.avg_price) + float(record.adr) * 2.0,2) if transaction.entry_type == 'COMPRA' \
                    else round(float(transaction.avg_price) - float(record.adr) * 1.5,2)
                transaction.earning_losing_value = round((transaction.closing_price * spectrum) -
                                                         (first_entry_price *spectrum), 2)

            if not created:
                transaction.updated_at = datetime.now()

            transaction.save()

            if transaction.status == 'close' or idx == len(dataset[start_index:])-1:
                result['closed'] = transaction.status == 'close'
                result['stopped_at_index'] = dataset.index(record)
                result['last_record'] = record
                break

        result['status'] = True

    except Exception as X:
        result['status'] = False
        result['traceback'] = X.__traceback__
        result['err_obj'] = X
        result['error'] = f"{X}"

    # ()
    return result

# needs debugging
def calculate_tp_sl_on_records(start_date: datetime,
                               symbol: str, stored:bool) -> dict:
    """
    calculates the take profit values and stop loss on the records
    that matches the given description
    :param start_date: datetime: the date to start fetching data and calculating from
    :return: dict
    """
    result = {}
    existing_data = [r for r in Stock.objects.get(symbol=symbol).historicaldata_set.filter(api_date__gte=start_date.date())]
    last_transaction = HistoricalTransactionDetail.get_last_transaction(symbol=symbol)
    last_transaction_closed = last_transaction.status == 'open' if last_transaction else False
    start_index = 0
    finished = False

    try:
        while not finished:
            transaction_id = last_transaction.transaction_id + 1 if last_transaction_closed\
                else last_transaction.transaction_id if last_transaction else 1

            # transaction_id = HistoricalTransactionDetail.gen_transaction_id(symbol)

            if last_transaction_closed:

                # print(f"start index while finding pattern: {start_index}")

                start_at = find_concurrent_patterns(dataset=existing_data,
                                                    start_index=start_index)

                if not start_at['status'] and 'error' in start_at:
                    raise Exception(start_at['error'])

                elif not start_at['status']:
                    finished = True
                    continue
                start_index = start_at['start_index']

            # print(f"START INDEX BEFORE CALCULATION: {start_index}, LEN :{len(existing_data[start_index:])}")
            tp_sl_result = _calculate_tp_sl(dataset=existing_data,
                                            transaction_id=transaction_id,
                                            start_index=start_index,
                                            stored=stored
                                            )

            if not tp_sl_result['status']:
                raise Exception(tp_sl_result['error'])

            # need to check whether the transaction is closed or not
            # if closed, I need to be able restart the whole process.
            # if tp_sl_result['closed']:
            start_index = tp_sl_result['stopped_at_index']
            last_transaction_closed = tp_sl_result['closed']
            last_transaction = tp_sl_result['last_record'].historicaltransactiondetail
            finished = tp_sl_result['last_record'] == existing_data[-1]
            # print(f"START INDEX AFTER CALCULATION: {start_index}, LEN: {len(existing_data[start_index:])}")
            # print(tp_sl_result, finished, symbol, transaction_id, "END TRANSACTION HERE")

        result['status'] = True

    except Exception as X:
        result['status'] = False
        result['traceback'] = X.__traceback__
        result['err_obj'] = X
        result['error'] = f"{X}"

    # ()
    return result


def fetch_markets_data(symbols:list, interval: str='1h') -> dict:
    """
    fetches the data for all of the markets in the list based on the needed and then
    stores the data in the data base as needed for further processing
    :param symbols: list of symbols required for the execution
    :param start_date: datetime: when to start fetching from
    :param end_date: last time to stop fetching from
    :param interval: interval to get the data from :
    for now 1hour but it supports daily, monthly, 1min, 15min, etc.
    :return:dict
    """
    result = {'error': "",
              'status': False
              }
    window = 30

    try:
        dates = generate_time_intervals_for_api_query()
        print(dates)
        if not dates['status']:
            raise Exception(dates['error'])

        start_date = dates['start_date']-timedelta(days=1) if dates['start_date'].date() == datetime.today().date() \
                        and (dates['start_date'].hour > 18 or dates['start_date'].hour < 9) else dates['start_date']

        end_date = dates['end_date']

        for symbol in symbols:
            status, api_data = DATA_API_OBJ.live_market_data_getter(symbol=symbol, start_date=start_date,
                                                                    end_date=end_date, interval=interval)
            if status:
                data_container = {}
                stored = False
                stock_details = api_data['stock_details']
                data_container['api_data'] = api_data['data']
                #
                if(start_date.year == end_date.year and
                       Stock.listed_last_historical_data_fetch(symbol=symbol, refresh_time=start_date)):
                    stored = True
                    #
                    # obj = Stock.objects.get(symbol=symbol)

                    # here's the problem with the time formatting.
                    records = get_last_records(symbol=symbol,
                                               first_record_date=data_container['api_data'][0]['datetime'],
                                                                    window_number=window)
                    print(status, symbol, dates)
                    if not records['status']:
                        raise  Exception(records['error'])
                    data_container['model_data'] = records['data']
                    # data_container['model_data'] = HistoricalData.objects.filter(stock=obj.id).order_by('-api_date')[:14]
                    # data_container['model_data'] = [record for record in data_container['model_data']]
                    # data_container['model_data'].reverse()

                # we got an issue here
                operations = generate_statistical_indicators(data=data_container, stored=stored,
                                                             symbol=symbol,
                                                             start_date=start_date)
                if not operations['status']:
                    raise Exception(operations['error'])

                status, error = store_full_data(stock_details, operations,
                                                update=stored)

                result['status'] = status
                result['start_date'] = start_date

                if not status:
                    raise Exception(error)

                print(f"Data for symbol: {symbol}, has been successfully added to the DB"
                          f" with the intervals {start_date} to {end_date}.")
            else:
                raise Exception(api_data)

    except Exception as X:
        result['status'] = False
        result['error'] = f"{X}"
        result['traceback'] = X.__traceback__
        # ()

    finally:
        return result


def calculate_transactions_for_symbols(start_date:datetime, symbols:list)->dict:
    """
    calculates the transactions for all given symbols
    based on the start_date
    :param start_date: datetime
    :param symbols: list
    :return: dict
    """
    stored = True
    result = {}
    try:
        for symbol in symbols:
            sl_tp_calculation = calculate_tp_sl_on_records(start_date=start_date,
                                                           symbol=symbol,
                                                           stored=stored)
            if not sl_tp_calculation['status']:
                raise Exception(sl_tp_calculation['error'])

        result['status'] = True
        result['message'] = f'Transaction details Successfully generated from {start_date.isoformat()}'
    except Exception as X:
        result['status'] = False
        result['error'] = f"{X}"
        result['traceback'] = X.__traceback__
        # ()

    return result

def stock_excel_loader(excel_file:str, sheet_number:int=2):
    """
    Loads the values of the excel file to
    the data base for further processing.
    :param excel_file: text wrapper I/O
    :param sheet_number: number of sheet to open
    :return: dict
    """
    result = {'status': False}
    try:
        f = open(excel_file,'rb')
        excel_data = pd.read_excel(f, sheet_name=sheet_number).to_dict()
        high_priority = ['AMZN','FB','AAPL','NVDA','TSLA','MSFT','GOOGL']
        for k in excel_data['INDEX'].keys():
            priority = Stock.choices[0] if excel_data['SYMBOL'][k] in high_priority else Stock.choices[2]
            stock, created = Stock.objects.update_or_create(symbol=excel_data['SYMBOL'][k],
                                                            priority=priority)
            if created:
                stock.stock_details = json.dumps({'name': excel_data['STOCK NAME'][k]})
                stock.save()

            else:
                details = json.loads(stock.stock_details)
                if details:
                    details.update(name=excel_data['STOCK NAME'][k])
                else:
                    details = {'name': excel_data['STOCK NAME'][k]}

                stock.stock_details = json.dumps(details)
                stock.save()

        result['status'] = True
    except Exception as X:
        result['error'] = f"There was an error with your request {X}"

    return result


def refresh_stock_list():
    """
    refreshes the list of stocks in the db
    :return: dict
    """
    result = {}
    try:
        stocks = DATA_API_OBJ.load_markets()
        high_priority = open(f"{settings.BASE_DIR}/high_priority_stocks.csv",'r'
                              ).read().split(',')
        for stock in stocks:
            s, created = Stock.objects.get_or_create(
                symbol=stock['symbol'],
            )
            s.priority = Stock.choices[0] if stock['symbol'] in high_priority else Stock.choices[1]
            s.name = stock['name']
            s.stock_details = json.dumps(stock)
            print(s.priority)
            if not created:
                s.updated_at = datetime.now()
            s.save()
        result['status'] = True

    except Exception as X:
        result['status'] = False
        result['error'] = f"{X}"
        result['traceback'] = X.__traceback__
        # ()

    return result

def load_high_priority_stocks_by_file(file:str=f"{settings.BASE_DIR}/high_priority_stocks.csv"):
    """
    loads the stocks with highest priority
    by reading the file and just updloading them to the DB
    :param file: path of the file
    :return: dict
    """
    result = {}
    try:

        stocks = [r.replace(' ','') for r in open(file,'r').read().replace('\n','').split(',')]
        print(stocks)
        for stock in stocks:
            s, created = Stock.objects.get_or_create(
                symbol=stock
            )
            s.priority = Stock.choices[0]
            if not created:
                s.updated_at = datetime.now()
            s.save()
        result['status'] = True

    except Exception as X:
        result['status'] = False
        result['error'] = f"{X}"
        result['traceback'] = X.__traceback__
        # ()

    return result
