from datetime import datetime, timedelta
from.DataFetcherService import  *
# import statistics as stats
import plotly.graph_objects as p_go
from plotly.offline import iplot
# import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import pdb
import json
from .models import *
import calendar


DATA_API_OBJ = APIDataHandler()

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

# def generate_definite_graph(data: dict, stock_details: dict):
#     """
#     generates the definite graph needed in order to manage the values required
#     and provide a final graph
#     :param data: dict
#     :param stock_details: dict
#     :return: plotly graph
#     """
#
#
#     status, calendar = DATA_API_OBJ.get_market_calendar("2020")
#     # pdb.set_trace()
#     graph = make_subplots(rows=5, cols=1)
#     for option in data.keys():
#         df = data[option]
#
#         if option == 'olhcv':
#             inner_graph = p_go.Candlestick(x=df['datetime'],
#                                            open=df['open'], high=df['high'],
#                                            low=df['low'], close=df['close'])
#             graph.append_trace(inner_graph, 1, 1)
#             # graph.add_trace(p_go.Bar(x=df['datetime'], y=df['volume']),
#             #                 secondary_y=False)
#             graph.update_traces(name='OHLC', selector=dict(type='candlestick'))
#             # graph.update_traces(name='Volume', selector=dict(type='bar'), opacity=0.70)
#             # graph.update_layout(height=750)
#             graph.update_xaxes(
#                 rangeslider_visible=True,
#                 rangebreaks=[
#                     # NOTE: Below values are bound (not single values), ie. hide x to y
#                     dict(bounds=["sat", "mon"]),
#                     dict(bounds=[22, 6], pattern="hour"),
#                     # dict(values=["2020-12-25", "2021-01-01"])  # hide holidays (Christmas and New Year's, etc)
#                 ]
#             )
#             # graph.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
#
#
#         elif option == 'rsi':
#             graph.append_trace(p_go.Scatter(y=df['datetime'], x=df['operation_data']), 2, 1)
#             graph.update_traces(name='RSI', showlegend=True, selector=dict(type='scatter'))
#             # graph.update_layout(height=300)
#
#         elif option == 'stochastic':
#             graph.append_trace(p_go.Scatter(x=df['datetime'], y=df['k_fast'],
#                                          y0=df['d_slow'],
#                                          name='k_fast'), 3, 1)
#             graph.update_traces(name='K_FAST', showlegend=True, selector=dict(type='scatter',
#                                                                               name='k_fast'), line_color='green')
#
#             graph.add_trace(p_go.Scatter(x=df['datetime'], y=df['k_slow'],
#                                          name='k_slow'))
#             graph.update_traces(name='K_SLOW', showlegend=True, selector=dict(type='scatter',
#                                                                               name='k_slow'), line_color='magenta')
#             # graph.update_layout(height=300)
#
#
#         elif option == 'macd':
#             graph.append_trace(p_go.Scatter(x=df['datetime'], y=df['macd'],
#                                          name='MACD'), 4, 1)
#             graph.update_traces(name='MACD', showlegend=True, selector=dict(type='scatter',
#                                                                             name='MACD'),
#                                 line_color='red')
#             graph.add_trace(p_go.Scatter(x=df['datetime'], y=df['signal'],
#                                          name='Signal Line'))
#
#             graph.update_traces(name='Signal Line', showlegend=True,
#                                 selector=dict(type='scatter',
#                                               name='Signal Line'), line_color='purple')
#
#             # graph.update_layout(height=300)
#
#         elif option == 'adr':
#             graph.append_trace(p_go.Scatter(y=df['datetime'], x=df['operation_data'],
#                                          name='adr'), 5, 1)
#             graph.update_traces(name='adr', showlegend=True, selector=dict(type='scatter',
#                                                                            name='adr'),
#                                 line_color='orange')
#
#             # graph.update_layout(height=300)
#
#     graph.update_layout(
#         title=f'{stock_details["symbol"]} Market Graphs',
#         yaxis_title=f'{stock_details["Code"]} Stock',
#         legend=dict(
#             yanchor="top",
#             y=0.999,
#             xanchor="left",
#             x=-0.05
#         )
#     )
#
#     graph.update_layout(xaxis_rangeslider_visible=False)
#
#         # graph.layout.yaxis2.showgrid = True
#
#     return graph
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
    # pdb.set_trace()
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
            # pdb.set_trace()
            graph.update_traces(name=option.title(), selector=dict(type='scatter'))
            result = graph
            status = True

        elif option == 'stochastic':
            graph.add_trace(p_go.Scatter(x=new_values['datetime'], y=new_values['d_fast']),
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
    result = pd.DataFrame(data={'macd': macd, 'signal':signal,'datetime':data['datetime']})
    return result



def calculate_adr(dt: pd.DataFrame) -> pd.DataFrame:
    """
    generates the calulation for the ADR
    :param data:
    :return: pandas DataFrame
    """
    data = dt.copy()
    high = data['high'].astype(dtype=float)
    low = data['low'].astype(dtype=float)
    close = data['close'].astype(dtype=float)
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    adr = lambda values, n: values.ewm(alpha=1/n, adjust=False).mean()
    return adr(tr,14)



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
    # rsi = rsi.dropna()
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
    stochastic['k_fast'] = 100 * (stochastic['close'].astype(dtype=float) -
                                  low_min.astype(dtype=float)) / (high_max.astype(dtype=float) -
                                                                  low_min.astype(dtype=float))
    stochastic['d_fast'] = stochastic['k_fast'].rolling(window=d).mean()

    # Slow Stochastic
    stochastic['k_slow'] = stochastic["d_fast"]
    stochastic['d_slow'] = stochastic['k_slow'].rolling(window=d).mean()

    return stochastic


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
            # pdb.set_trace()
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
               operation_data:dict) -> tuple:
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

        for i in range(len(operation_data['olhcv'])):
            if evaluate_integrity(operation_data['olhcv'].iloc[i],operation_data['rsi'].iloc[i],
                                  operation_data['adr'].iloc[i],operation_data['macd'].iloc[i],
                                  operation_data['stochastic'].iloc[i]):

                f_stoch = 'HIGH' if operation_data['stochastic'].iloc[i]['k_fast'] > 20.00 else 'LOW'
                f_rsi = 'HIGH' if operation_data['rsi'].iloc[i]['operation_data'] > 70.00 else 'LOW'
                f_macd = 'HIGH' if operation_data['macd'].iloc[i]['macd'] > operation_data['macd'].iloc[i]['signal'] \
                    else 'LOW'
                bullet = 'START' if f_stoch == f_rsi == f_macd == 'HIGH' else 'STOP' \
                    if f_stoch == f_rsi == f_macd == 'LOW' else 'PAUSE'

                HistoricalData.objects.update_or_create(
                                                stock = stock,
                                                open = operation_data['olhcv'].iloc[i]['open'],
                                                high = operation_data['olhcv'].iloc[i]['high'],
                                                low = operation_data['olhcv'].iloc[i]['low'],
                                                close = operation_data['olhcv'].iloc[i]['close'],
                                                volume = operation_data['olhcv'].iloc[i]['volume'],
                                                api_date = operation_data['olhcv'].iloc[i]['datetime'],
                                                rsi = operation_data['rsi'].iloc[i]['operation_data'],
                                                adr = operation_data['adr'].iloc[i]['operation_data'],
                                                k_slow = operation_data['stochastic'].iloc[i]['k_slow'],
                                                k_fast = operation_data['stochastic'].iloc[i]['k_fast'],
                                                macd = operation_data['macd'].iloc[i]['macd'],
                                                signal = operation_data['macd'].iloc[i]['signal'],
                                                f_stoch=f_stoch,
                                                f_rsi = f_rsi,
                                                f_macd = f_macd,
                                                bullet= bullet
                                                )
                # StochasticIndicator.objects.create(historical_data=historical,
                #                                    k_slow=operation_data['stochastic'].iloc[i]['k_slow'],
                #                                    k_fast=operation_data['stochastic'].iloc[i]['k_fast'],
                #                                    api_date=operation_data['stochastic'].iloc[i]['datetime']
                #                                    )
                # MACDIndicator.objects.create(
                #     historical_data=historical,
                #     macd=operation_data['macd'].iloc[i]['macd'],
                #     signal=operation_data['macd'].iloc[i]['signal'],
                #     api_date=operation_data['macd'].iloc[i]['datetime']
                # )

        status = True
    except Exception as X:
        error = f"There was an error with the request: {X}"
        print("ERROR HERE", X)

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
        # pdb.set_trace()
        # datetime.sleep(20)
    return pd.Series(data={k:v for k, v in enumerate(serialized)})


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
        # pdb.set_trace()


        if status:
            stock_details = api_data['stock_details']
            # proccesed_data = serialize_data(pd.DataFrame(api_data['data']).dropna())
            proccesed_data = pd.DataFrame(api_data['data']).dropna()
            # print(proccessed_data)
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
            # pdb.set_trace()
            status, error = store_full_data(stock_details, operations)
            if not status:
                return status, error
            # for key in operations:
            #     status, error = store_data(stock_details,operations[key],key)
            #     if not status:
            #         return status, error
            main_graph = generate_live_graph(proccesed_data, stock_details)
            rsi_graph = generate_live_graph(stock_data=operations['rsi'],stock_details=stock_details, option="rsi")
            stochastic_graph = generate_live_graph(stock_data=operations['stochastic'],
                                                   stock_details=stock_details, option="stochastic")
            macd_graph = generate_live_graph(stock_data=operations['macd'],
                                             stock_details=stock_details,
                                       option='macd')
            adr_graph = generate_live_graph(stock_data=operations['adr'],stock_details=stock_details,option='adr')

            # graph = generate_definite_graph(data=operations,
            #                                 stock_details=stock_details)
            result = {
                # 'graph': render_graph(graph),
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

