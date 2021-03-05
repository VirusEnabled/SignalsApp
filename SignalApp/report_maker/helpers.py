from datetime import datetime, timedelta
from.TradierDataFetcherService import  *
# import statistics as stats
import plotly.graph_objects as p_go
# import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import pdb
import json
from .models import *


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

# need to remove the gaps between the graphs based on what's needed
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

    df = stock_data
    graph =  make_subplots(specs=[[{"secondary_y": True}]])

    if option == 'olhcv':
        inner_graph = p_go.Candlestick(x=df['time'],
                                       open=df['open'], high=df['high'],
                                       low=df['low'], close=df['close'])
        graph.add_trace(inner_graph,
                        secondary_y=True)
        graph.add_trace(p_go.Bar(x=df['time'], y=df['volume']),
                        secondary_y=False)
        graph.update_traces(name='OHLC', selector=dict(type='candlestick'))
        graph.update_traces(name='Volume', selector=dict(type='bar'), opacity=0.70)
        graph.update_layout(height=750)
        graph.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["fri", "mon"]),  # hide weekends, eg. hide sat to before mon
                dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                # dict(values=["2020-12-25", "2021-01-01"])  # hide holidays (Christmas and New Year's, etc)
            ]
        )
        # graph.update_xaxes(rangebreaks=[dict(values=dt_breaks)])


    elif option == 'rsi':
        graph.add_trace(p_go.Scatter(y=df['time'], x=df['operation_data']),
                        secondary_y=True)
        graph.update_traces(name='RSI',showlegend=True ,selector=dict(type='scatter'))
        graph.update_layout(height=300)

    elif option == 'stochastic':
        graph.add_trace(p_go.Scatter(x=df['time'], y=df['k_fast'],
                                     y0=df['d_slow'],
                                     name='k_fast'),
                        secondary_y=True)
        graph.update_traces(name='K_FAST',showlegend=True ,selector=dict(type='scatter',
                                                                         name='k_fast'),line_color='green')

        graph.add_trace(p_go.Scatter(x=df['time'], y=df['k_slow'],
                                     name='k_slow'),
                        secondary_y=True)
        graph.update_traces(name='K_SLOW', showlegend=True, selector=dict(type='scatter',
                                                                          name='k_slow'), line_color='magenta')
        graph.update_layout(height=300)


    elif option == 'macd':
        pass
        # plt.plot(df.ds, macd, label='AMD MACD', color='#EBD2BE')
        # plt.plot(df.ds, exp3, label='Signal Line', color='#E5A4CB')
        graph.add_trace(p_go.Scatter(x=df['time'], y=df['macd'],
                                     name='MACD'),
                        secondary_y=True)
        graph.update_traces(name='MACD', showlegend=True, selector=dict(type='scatter',
                                                                          name='MACD'),
                            line_color='red')
        graph.add_trace(p_go.Scatter(x=df['time'], y=df['signal'],
                                     name='Signal Line'),
                        secondary_y=True)

        graph.update_traces(name='Signal Line', showlegend=True,
                            selector=dict(type='scatter',
                                          name='Signal Line'), line_color='purple')

        graph.update_layout(height=300)


    graph.update_layout(
        title=f'{stock_details["Name"]} Market: {option.capitalize()} graph',
        yaxis_title=f'{stock_details["Code"]} Stock',
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
            # pdb.set_trace()
            graph.update_traces(name=option.title(), selector=dict(type='scatter'))
            result = graph
            status = True

        elif option == 'stochastic':
            graph.add_trace(p_go.Scatter(x=new_values['time'], y=new_values['d_fast']),
                            secondary_y=False)
            # pdb.set_trace()
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


def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    """
    generates the calulation for the MACD
    :param data:
    :return: dataframe
    """
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    result = pd.DataFrame(data={'macd': macd, 'signal':signal,'time':data['time']})
    return result



def calculate_adr(data: list) -> pd.DataFrame:
    """
    generates the calulation for the ADR
    :param data:
    :return:
    """

def calculate_rsi(data: pd.DataFrame) -> pd.Series:
    """
    generates the calulation for the RSI
    :param data:
    :return: float
    """

    window_length = 14  # this should change based on the needs
    delta = data['close'].astype(dtype=float).diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.dropna().clip(lower=1)
    # print(rsi)
    return rsi


def calculate_stochastic(data: pd.DataFrame) -> pd.DataFrame:
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

    """
    stochastic = data.copy().dropna()
    d= 3 # 3 days of SMA which come from K
    k = 14 # days of the window
    low_min = stochastic['low'].rolling(window=k).min().dropna()
    high_max = stochastic['high'].rolling(window=k).max().dropna()

    # Fast Stochastic
    # pdb.set_trace()
    stochastic['k_fast'] = 100 * (stochastic['close'].astype(dtype=float) - low_min) / (high_max - low_min)
    stochastic['d_fast'] = stochastic['k_fast'].rolling(window=d).mean()

    # Slow Stochastic
    stochastic['k_slow'] = stochastic["d_fast"]
    stochastic['d_slow'] = stochastic['k_slow'].rolling(window=d).mean()

    return stochastic



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

def to_data_frame(operation_data, time_data):
    """
    creates a dataframe to plot the results of the calculations
    :param operation_data: data used to calculate the indicators
    :param time_data: time elapsed
    :return: dataframe object
    """
    dataframe = pd.DataFrame()
    dataframe['time'] = operation_data
    dataframe['operation_data'] = time_data
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
        stock = Stock.objects.get(symbol=stock_details['Code']) if Stock.exists(stock_details['Code'])\
            else Stock.objects.create(symbol=stock_details['Code'],
                                      stock_details=json.dumps(stock_details))
        stock.indicatorcalculationdata_set.create(indicator=operation.capitalize(),
                                                  operation_data=operation_data.to_json())
        status = True
    except Exception as X:
        error = f"There was an error with the request: {X}"
        print("ERROR HERE", X)

    finally:
        return status, error

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
    :param interval : str : Interval of time per timesale. One of: tick, 1min, 5min, 15min
    :return: tuple
    """
    status, result = False, {'error':""}
    try:
        # status, adr = calculate_adr()
        status, api_data = TRADIER_API_OBJ.live_market_data_getter(symbol=symbol, start_date=start_date,
                                                           end_date=end_date,interval=interval)
        st, stock_details = load_company_details(symbol=symbol)

        if status and st:
            proccessed_data = pd.DataFrame(api_data).dropna()
            # pdb.set_trace()
            print(proccessed_data)
            operations = dict(stochastic=calculate_stochastic(data=proccessed_data),
                              rsi=to_data_frame(operation_data=calculate_rsi(pd.DataFrame(api_data)),
                                time_data=proccessed_data['time']),
                              macd=calculate_macd(proccessed_data))

            for key in operations:
                status, error = store_data(stock_details,operations[key],key)
                if not status:
                    return status, error

            main_graph = generate_live_graph(proccessed_data, stock_details)
            rsi_graph = generate_live_graph(stock_data=operations['rsi'],stock_details=stock_details, option="rsi")
            stochastic_graph = generate_live_graph(stock_data=operations['stochastic'],
                                                   stock_details=stock_details, option="stochastic")
            macd_graph = generate_live_graph(stock_data=operations['macd'],
                                             stock_details=stock_details,
                                       option='macd')

            result = {
                'graph': render_graph(main_graph),
                'rsi_graph': render_graph(rsi_graph),
                'stochastic_graph': render_graph(stochastic_graph),
                'macd_graph': render_graph(macd_graph)

            }
        else:
            status = False
            result = api_data if not status else stock_details

    except Exception as X:
        result['error'] = f"There has been an error with your request: {X}"
        status = False
        print(status,result,"EXITING IT")

    finally:
        return status, result

