import requests as req
import hmac
import hashlib
import json
from tradier import tradier
from datetime import datetime, date, time, timedelta

class TradierDataHandler(object):

    def __init__(self):
        self.config = {
            'sandbox_credentials':{
                'account_number':"VA67273733",
                'access_token': "rpxricq2EY5WbhosGLM8tmP8lTPW"
            },
            'broker_token':"qglG9G87oKaSYgDm2TWtzkiFEtKu",
            'historical_data_api_eod': "60285a53993948.89589431"

        }
        self.api_root_endpoints = {
            'brokerage_rest':"https://api.tradier.com/v1/",
            'brokerage_stream': "https://stream.tradier.com/v1/",
            'sandbox':"https://sandbox.tradier.com/v1/",
            'eod_historical_data': "https://eodhistoricaldata.com/api/"

        }

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
            'Authorization': f"Bearer {self.config['broker_token']}",
            'Content-Type':'application/json'
        }

    def refresh_access_token(self) -> None :
        """
        refreshes the existing access token
        so that there's no isue with the
        security of the site.
        :return: None
        """
        endpoint = f"{self.api_root_endpoints['brokerage_rest']}oauth/refreshtoken"

    def get_quotes(self):
        """
        gets all of the available quotes
         for the existing stocks
        :return: list
        """
        endpoint = f"{self.api_root_endpoints['brokerage_rest']}markets/quotes"

    def get_quote(self, symbol: str) -> dict:
        """
        gets the quotes values for the given stock
        :param symbol: str
        :return: dict
        """
        endpoint = f"{self.api_root_endpoints['brokerage_rest']}markets/quotes"


    def load_exchanges(self) -> list:
        """
        lists all of the available exchanges
        :return: list
        """
        exchanges_endpoint = f"{self.api_root_endpoints['eod_historical_data']}exchanges-list/?api_token={self.config['historical_data_api_eod']}&fmt=json"
        response =  req.get(exchanges_endpoint)
        return response.json()

    def _markets(self):
        """
        loads the values from the API into a file
        :return: None
        """
        with open('markets.json','w+') as market_file:
            market_file.write("[")
            for market in self.markets:
                market_file.write(f"{json.dumps(market)},\n")
            market_file.write("]")
        market_file.close()

    def load_markets(self):
        """
        loads the values from
        the market file object
        :return: list
        """
        return json.loads(open('markets.json','r').read())

    @property
    def markets(self) -> list:
        """
        gets all of the existing symbols
        required for building the queries required
        in order to get historical data
        for now just USA as requested by the client.
        :return: list
        """
        # exchanges = self.load_exchanges()
        endpoint =  f"{self.api_root_endpoints['eod_historical_data']}exchange-symbol-list/US?api_token={self.config['historical_data_api_eod']}&fmt=json"
        response = req.get(endpoint)
        return response.json()


    def get_historical_data(self, symbol: str, interval: str =None,
                            start_date: date =None,
                            end_date: date =None) -> dict:
        """
        gets the historical values on the existing
        stock for different markers olhcv
        :param symbol: str: symbol that represents the stock
        :param interval: str: when the data coming from, only options: daily, weekly, monthly
        :param start_date: date: when it should start parsing from: YYYY-MM-DD
        :param end_date: date: where it should stop on: YYYY-MM-DD
        :return: dict
        """
        endpoint = f"{self.api_root_endpoints['brokerage_rest']}markets/history"
        params = {'symbol': symbol,
                  'interval': 'daily' if not interval else interval
                  }
        if start_date and end_date:
            params['start'] = start_date.strftime("%YYYY-%MM-%DD")
            params['end'] = end_date.strftime("%YYYY-%MM-%DD")

        response = req.get(endpoint, params=params, headers=self._headers)
        # print(response, response.headers)
        return response.json()


    def get_all_historical_data_info(self, start_date: date =None,
                            end_date: date =None) -> tuple:
        """
        loads all of the historical data
        for all stocks based on the dates given
        :return: tuple
        """
        flag =  False
        historical_data = None
        markets = self.load_markets()

        try:
            available_symbols = [obj['Code'] for obj in markets]
            historical_data = [
                self.get_historical_data(symbol, start_date=start_date, end_date=end_date)
                for symbol in available_symbols
            ]
        except Exception as X:
            historical_data = f"There was an error{X}"

        finally:
            return flag, historical_data