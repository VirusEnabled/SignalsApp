import requests as req
import json
from datetime import datetime, date, time, timedelta
from django.conf import settings

class APIDataHandler(object):

    def __init__(self):
        self.config = {
            'sandbox_credentials': {
                'account_number':"VA67273733",
                'access_token': "rpxricq2EY5WbhosGLM8tmP8lTPW"
            },
            'broker_token': "qglG9G87oKaSYgDm2TWtzkiFEtKu",
            'historical_data_api_eod': "60285a53993948.89589431",
            'twelve_api_key': "eb61c42448454dc5b6b6f59dfe6d8072",

        }
        self.api_root_endpoints = {
            'brokerage_rest': "https://api.tradier.com/v1/",
            'brokerage_stream': "https://stream.tradier.com/v1/",
            'sandbox': "https://sandbox.tradier.com/v1/",
            'eod_historical_data': "https://eodhistoricaldata.com/api/",
            'twelve_api': "https://api.twelvedata.com/",

        }
        # self.save_market_list()

    @property
    def _headers(self) -> dict:
        """
        builds the authentication
        headers needed in order
        to access the API
        :return: dict
        """
        return {
            'Accept':"application/json",
            # 'Authorization': f"Bearer {self.config['broker_token']}",
            'Content-Type':'application/json'
        }

    # def refresh_access_token(self) -> None :
    #     """
    #     refreshes the existing access token
    #     so that there's no isue with the
    #     security of the site.
    #     :return: None
    #     """
    #     endpoint = f"{self.api_root_endpoints['brokerage_rest']}oauth/refreshtoken"
    #
    # def get_quotes(self):
    #     """
    #     gets all of the available quotes
    #      for the existing stocks
    #     :return: list
    #     """
    #     endpoint = f"{self.api_root_endpoints['brokerage_rest']}markets/quotes"
    #
    # def get_quote(self, symbol: str) -> dict:
    #     """
    #     gets the quotes values for the given stock
    #     :param symbol: str
    #     :return: dict
    #     """
    #     endpoint = f"{self.api_root_endpoints['brokerage_rest']}markets/quotes"


    def load_exchanges(self) -> list:
        """
        lists all of the available exchanges
        :return: list
        """
        exchanges_endpoint = f"{self.api_root_endpoints['eod_historical_data']}exchanges-list/?api_token={self.config['historical_data_api_eod']}&fmt=json"
        response =  req.get(exchanges_endpoint)
        return response.json()


    @classmethod
    def load_markets(cls):
        """
        loads the values from
        the market file object
        :return: list
        """
        result = []
        endpoint = f"https://api.twelvedata.com/stocks"
        # headers= cls._headers
        # import pdb;pdb.set_trace()
        flag, value = settings.REDIS_OBJ.get_item('stock_list')
        if flag:
            result = value
        else:
            response = req.get(endpoint)
            settings.REDIS_OBJ.load_value('stock_list', response.json()['data'])
            result = response.json()['data']

        return result

    @property
    def markets(self) -> list:
        """
        gets all of the existing symbols
        required for building the queries required
        in order to get historical data
        for now just USA as requested by the client.
        :return: list
        """
        return self.load_markets()


    def live_market_data_getter(self, symbol:str , interval:str =None,
                                start_date: datetime=datetime.today()-timedelta(days=365),
                                end_date: datetime=datetime.today()) -> tuple:
        """
        get the data coming from the markets
        this data should be live data meaning
        actual data coming in based on the given symbol
        :param symbol: str symbol needed
        :param interval : str : Interval of time per timesale. One of: tick, 1min, 5min, 15min
        :param start_date: datetime: when to start parsing from
        :param end_date: datetime: when to stop parsing to
        :return: tuple
        """
        flag = False
        result = {}
        try:
            endpoint = f"{self.api_root_endpoints['twelve_api']}time_series"
            params = {'symbol': symbol,
                      'interval': '1h' if not interval else interval,
                      'start_date ': start_date.date().isoformat(),
                      'end_date': end_date.date().isoformat(),
                      'timezone':'America/New_York',
                      'apikey': self.config['twelve_api_key'],
                      'order':"asc",
                      }
            # start_date = start_date - timedelta(days=1) if start_date.date() == datetime.now().date() else start_date
            # definitive_endpoint = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&" \
            #                       f"apikey={self.config['twelve_api_key']}" \
            #                       f"&source=docs&start_date={start_date.date().isoformat()}"

            definitive_endpoint = f"https://api.twelvedata.com/stoch?symbol={symbol}&interval=1h&" \
                                  f"apikey={self.config['twelve_api_key']}" \
                                  f"&source=docs&start_date={start_date.date().isoformat()}&" \
                                  f"include_ohlc=true&slow_k_period=3&fast_k_period=14"

            print(definitive_endpoint, start_date.date().isoformat())
            response = req.get(definitive_endpoint)
            # response = req.get(endpoint, params=params, headers=self._headers)
            response_data = response.json()
            print(response_data.keys())
            if 'values' not in response_data.keys():
                result = {'error': f"The data related to the symbol: {symbol} was not found, try a different one \n "
                                   f"{response_data['message']}"}
                flag=False

            else:

                flag = True
                print(f"RECORDS: {len(response_data['values'])}")
                result = {'stock_details':response_data['meta'],
                          "data": [
                              {'open':record['open'],
                               'high': record['high'],
                               'low': record['low'],
                               'close': record['close'],
                               'volume': record['volume'],
                               'k': record['slow_k'],
                               'datetime': record['datetime'],
                               }
                           for record in response_data['values']
                              # if float(record['volume']) > 0.00
                              ]
                          }
                result['data'].reverse()
                # result['data'] = [value for value in result['data'] if float(value['volume']) > 0.00]

        except json.JSONDecodeError as JS:
            flag = False
            result = {'error': f"There was an error with the request: {response.text}, please try again later."}

        except Exception as X:
            flag = False
            result = {'error': f"There was an error with the request: {X}, please try again later."}

        finally:
            return flag, result

    def get_market_calendar(self, year):
        """
        gets the stock market calendar for the given year
        :param year: str
        :return: tuple
        """
        endpoint = f"{self.api_root_endpoints['brokerage_rest']}markets/calendar"
        flag = False
        result = {}
        final = []
        try:
            for month in range(1,13):
                params = {'year': year,
                          'month': f"{month}"
                          }

                response = req.get(endpoint, params=params, headers=self._headers)
                # print(response.text)
                if not response.json()['calendar']:
                    result = {'error': f"The year: {year} was not capable of providing data, try a different one"}
                else:
                    flag = True
                    result = response.json()['calendar']
                    final.append(result['calendar'])
            else:
                result = final

        except Exception as X:
            flag = False
            result = {'error': f"There was an error with the request: {X}, please try again later."}

        finally:
            return flag, result


    def get_historical_data(self, symbol: str, interval: str =None,
                            start_date: str =None,
                            end_date: str =None) -> tuple:
        """
        gets the historical values on the existing
        stock for different markers olhcv
        :param symbol: str: symbol that represents the stock
        :param interval: str: when the data coming from, only options: daily, weekly, monthly
        :param start_date: str: when it should start parsing from: YYYY-MM-DD
        :param end_date: str: where it should stop on: YYYY-MM-DD
        :return: tuple: bool and dict
        """
        flag = False
        result = {}
        try:
            endpoint = f"{self.api_root_endpoints['twelve_api']}time_series"
            params = {'symbol': symbol,
                      'interval': '1month' if not interval else interval
                      }
            if start_date and end_date:
                params['start_date'] = start_date
                params['end_date'] = end_date

            response = req.get(endpoint, params=params, headers=self._headers)
            # print(response.text)
            if not response.json()['values']:
                result = {'error': f"The data related to the symbol: {symbol} was not found, try a different one"}
            else:
                flag = True
                result = response.json()['values']

        except Exception as X:
            flag = False
            result = {'error': f"There was an error with the request: {X}, please try again later."}

        finally:
            return flag, result

    def get_all_historical_data_info(self, start_date: date,
                                     symbols: list,
                                     indicators: list) -> dict:
        """
        gets all of the data related to the given symbols
        and indicators
        :param start_date: when to start parsing from
        :param symbols: list: list of symbols to fetch from
        :param indicators: list: list of indicators to get from.
        :return: dict of all items gathered or full or errors.
        """
        result = {}
        try:
            endpoint = f"{self.api_root_endpoints['twelve_api']}complex_data?" \
                       f"apikey={self.config['twelve_api_key']}&start_date={start_date.isoformat()}"
            request_body = {
                'symbols': symbols,
                'intervals':['1h'],
                'methods': ['time_series'] + indicators
            }
            response = req.post(endpoint, json=request_body, headers=self._headers)
            if isinstance(response.json()['data'], list):
                result['status'] = True
                result['data'] = response.json()['data']
            else:
                # and 'code' not in response.json()['data'].keys()
                raise Exception("".join(response.json()['data']))

        except Exception as X:
            result['status'] = False
            result['error'] = f"There was an error: {X}"

        return result



    def save_market_list(self):
        """
        saves the market list in the
        caching mechanism
        :return: None
        """
        status, result = settings.REDIS_OBJ.load_market_list(self.markets)
        if not status:
            raise Exception(result)