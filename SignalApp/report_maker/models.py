from django.db import models
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from datetime import datetime as t


class BaseModel(models.Model):
    class Meta:
        abstract = True
    created_at =  models.DateTimeField(auto_now=True)
    updated_at =  models.DateTimeField(auto_now=True)



class Stock(BaseModel):
    choices = [('HIGH', 'HIGH'),
               ('LOW', 'LOW'),
               ('MED', 'MEDIUM')
               ]
    symbol = models.CharField(max_length=100)
    stock_details = models.TextField(default="{}")
    priority = models.CharField(max_length=20,
                                choices=choices,
                                default=choices[0])

    @property
    def prty(self):
        """
        renders a pritable version of the stock priority
        :return:
        """
        return self.choices[0][1] if 'HIGH' in self.priority else self.choices[1][1] \
            if 'LOW' in self.priority else self.choices[2][1]

    def __str__(self):
        return f"<{self.symbol}>"

    @staticmethod
    def exists(symbol):
        """
        verifies if the given symbol is in the db
        :param symbol: str
        :return: bool
        """
        result = False
        try:
            stock = Stock.objects.get(symbol=symbol)
            result = True
        except models.ObjectDoesNotExist:
            pass
        finally:
            return result

    @staticmethod
    def listed_last_historical_data_fetch(symbol: str, refresh_time: t) -> bool:
        """
        validates if the last historical data register equals the
        given refresh time. this is done in order to make sure
        you're updating the values related to the given symbol.
        :param symbol: str
        :param refresh_time: datetime
        :return: bool
        """
        result = False
        try:
            stock = Stock.objects.get(symbol=symbol)
            v = stock.historicaldata_set.last().api_date
            result = v.month == refresh_time.month and v.year == refresh_time.year
        except Exception as X:
            print(f"DEBUGGING: There was an error boi: NOT STORED IN MODEL {X}")

        return result

    @staticmethod
    def has_historical_data(symbol: str) -> bool:
        """
        verifies if the db has records available
        :param symbol: str
        :param refresh_time: datetime
        :return: bool
        """
        result = False
        try:
            stock = Stock.objects.get(symbol=symbol)
            result = True if stock.historicaldata_set.last() else False
        except Exception as X:
            print(f"DEBUGGING: There was an error boi: NOT STORED IN MODEL {X}")
        return result

class HistoricalData(BaseModel):
    choices = [('HIGH', 'HIGH'),
               ('LOW','LOW'),
               ('EMPTY', 'EMPTY')]

    choices_bullet = [('START', 'START'),
                      ('STOP','STOP'),
                      ('PAUSE','PAUSE'),
                      ('EMPTY', 'EMPTY'),]

    stock = models.ForeignKey(Stock,on_delete=models.CASCADE)
    open = models.FloatField(default=0.00)
    high = models.FloatField(default=0.00)
    low = models.FloatField(default=0.00)
    close = models.FloatField(default=0.00)
    rsi = models.FloatField(default=0.00)
    adr = models.FloatField(default=0.00)
    volume = models.FloatField(default=0.00)
    k_slow = models.FloatField(default=0.00)
    k_fast = models.FloatField(default=0.00)
    k = models.FloatField(default=0.00)
    macd = models.FloatField(default=0.00)
    signal = models.FloatField(default=0.00)
    f_stoch = models.CharField(max_length=20, default=choices[-1])
    f_rsi = models.CharField(max_length=20, default=choices[-1])
    f_macd = models.CharField(max_length=20, default=choices[-1])
    bullet = models.CharField(max_length=20, default=choices_bullet[-1])
    api_date = models.DateTimeField()


    def __str__(self):
        return f"HistoricalData: <{self.stock.symbol}>"

    @property
    def o(self):
        return self.open

    @property
    def h(self):
        return self.high

    @property
    def l(self):
        return self.low

    @property
    def c(self):
        return self.close

    @property
    def v(self):
        return self.volume

    @property
    def datetime(self):
        return self.api_date.strftime("%Y-%m-%d %H:%m")




    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None, **kwargs):
        rounding_fields = ['open','low','high','close','volume','rsi','adr','k','macd','signal'
                           ]
        for field in rounding_fields:
            stringed = f"{self.__getattribute__(field)}"
            setattr(self, field, float(stringed[:stringed.find('.')+5]))


        return super().save(force_insert=force_insert,force_update=force_update,
                     using=using,update_fields=update_fields)


class StochasticIndicator(BaseModel):
    historical_data = models.OneToOneField(HistoricalData, on_delete=models.CASCADE)
    k_slow = models.FloatField(default=0.00)
    k_fast = models.FloatField(default=0.00)
    api_date = models.DateTimeField()


class MACDIndicator(BaseModel):
    historical_data = models.OneToOneField(HistoricalData, on_delete=models.CASCADE)
    macd = models.FloatField(default=0.00)
    signal = models.FloatField(default=0.00)
    api_date = models.DateTimeField()


class IndicatorCalculationData(BaseModel):
    indicators = [('STOCHASTIC','STOCHASTIC'),('RSI',"RSI"),
                  ('MACD','MACD'),('ADR','ADR'),]
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    indicator = models.CharField(choices=indicators,max_length=100)
    operation_data = models.TextField(default="{}")


    def __str__(self):
        return f"<{self.symbol}>"


class HistoricalTransactionDetail(BaseModel):
    historical_data = models.OneToOneField(HistoricalData, on_delete=models.CASCADE)
    id_market = models.IntegerField(default=0)
    status = models.CharField(max_length=100, default='open')
    stop_loss_price = models.FloatField(default=0.00)
    take_profit_price = models.FloatField(default=0.00)
    avg_price = models.FloatField(default=0.00)
    entry_type = models.CharField(max_length=100, default='')
    number_of_unities = models.IntegerField(default=0)
    entry_price = models.FloatField(null=True)
    closing_price = models.FloatField(null=True)
    transaction_id = models.IntegerField(null=True, default=1)
    earning_losing_value = models.FloatField(null=True)

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None, **kwargs):
        rounding_fields = ['stop_loss_price', 'take_profit_price', 'avg_price', 'entry_price', 'closing_price',
                           'earning_losing_value']
        for field in rounding_fields:
            value = self.__getattribute__(field)
            if not value:
                continue

            stringed = str(value)
            setattr(self, field, float(stringed[:stringed.find('.') + 5]))


        return super().save(force_insert=force_insert, force_update=force_update,
                            using=using, update_fields=update_fields)


    @classmethod
    def get_last_transaction(cls, symbol:str) -> object:
        """
        gets the last transaction by the given symbol
        :param symbol: str
        :return: object
        """
        stock = Stock.objects.get(symbol=symbol)
        records = HistoricalTransactionDetail.objects.filter(id_market=stock.id)
        last_transaction = records.last()
        return last_transaction

    # needs debugging
    @staticmethod
    def gen_transaction_id(symbol:str) -> int:
        """
        generates the transaction ID
        for a record based on the existing scores
        :return: int
        """
        last_transaction = HistoricalTransactionDetail.get_last_transaction(symbol=symbol)
        transaction_id = 1
        if last_transaction:
            if last_transaction.transaction_id:
                if last_transaction.status == 'close':
                    transaction_id = last_transaction.transaction_id  + 1

                else:
                    transaction_id = last_transaction.transaction_id

        return transaction_id

    @classmethod
    def get_open_values_from_open_transaction(cls, transaction_id:int, symbol:str):
        """
        gets the opening values from the given transaction as long
        as the transaction is open
        :param transaction_id: int
        :return: list
        """
        opens  = [t.historical_data.open for t in
            HistoricalTransactionDetail.objects.filter(transaction_id=transaction_id)
                  if t.historical_data.stock.symbol == symbol
                  ]
        return opens

class ConsolidatedTransactions(BaseModel):
    stock_id = models.IntegerField(default=0)
    net_profit = models.FloatField(default=0.00)
    gross_profit = models.FloatField(default=0.00)
    gross_loss = models.FloatField(default=0.00)
    profit_factor = models.FloatField(default=0.00)
    total_closed_trades = models.FloatField(default=0.00)
    total_open_trades = models.FloatField(default=0.00)
    number_winning_trades = models.FloatField(default=0.00)
    number_losing_trades = models.FloatField(default=0.00)
    percent_profitable = models.FloatField(default=0.00)
    average_trade = models.FloatField(default=0.00)
    average_winning_trade = models.FloatField(default=0.00)
    average_losing_trades = models.FloatField(default=0.00)
    ratio_avg_win_avg_loss = models.FloatField(default=0.00)
    largest_winning_trades = models.FloatField(default=0.00)
    largest_losing_trade = models.FloatField(default=0.00)
    avg_bars_winning = models.FloatField(default=0.00)
    avg_bars_losing = models.FloatField(default=0.00)