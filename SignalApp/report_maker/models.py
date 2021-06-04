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
            print(f"DEBUGGING: There was an error boi {X}")

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




    # def save(self, force_insert=False, force_update=False, using=None,
    #          update_fields=None, **kwargs):
    #     try:
    #         obj = HistoricalData.objects.get(stock=self.stock, api_date=self.api_date)
    #         if kwargs['create']:
    #             raise Exception("The object you're trying to create already exists.")
    #
    #         else:
    #             raise  models.ObjectDoesNotExist
    #
    #     except models.ObjectDoesNotExist:
    #         return super().save(force_insert=force_insert,force_update=force_update,
    #                      using=using,update_fields=update_fields)


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
    status = models.CharField(max_length=100, default='close')
    stop_loss_price = models.FloatField(default=0.00)
    take_profit_price = models.FloatField(default=0.00)
    avg_price = models.FloatField(default=0.00)
    entry_type = models.CharField(max_length=100, default='')
    number_of_entry = models.IntegerField(default=0)
    entry_price = models.FloatField(null=True)
    closing_price = models.FloatField(null=True)
    transaction_id = models.IntegerField(null=True, default=1)
    earning_losing_value = models.FloatField(null=True)



    @classmethod
    def get_last_transaction(cls, symbol:str) -> object:
        """
        gets the last transaction by the given symbol
        :param symbol: str
        :return: object
        """
        records = [record for record in HistoricalTransactionDetail.objects.all()
                   if record.historical_data.stock.symbol == symbol
                   ]
        ids = [record.id for record in records]
        # records = HistoricalData.objects.filter(stock=Stock.objects.get(symbol=symbol))
        last_transaction = records[ids.index(max(ids))] if records else None
        return last_transaction

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
                    print(last_transaction.status,"CLOSED HERE",
                          last_transaction.transaction_id, transaction_id)

                else:
                    print(last_transaction.status,"OPEN HERE")
                    transaction_id = last_transaction.transaction_id
        else:
            transaction_id = None
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