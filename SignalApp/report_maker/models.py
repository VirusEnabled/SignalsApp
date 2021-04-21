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


