from django.db import models
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token


class BaseModel(models.Model):
    class Meta:
        abstract = True
    created_at =  models.DateTimeField(auto_now=True)
    updated_at =  models.DateTimeField(auto_now=True)



class Stock(BaseModel):
    symbol = models.CharField(max_length=100)
    stock_details = models.TextField(default="{}")


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
            stock =Stock.objects.get(symbol=symbol)
            result = True

        except models.ObjectDoesNotExist:
            pass
        finally:
            return result



class HistoricalData(BaseModel):
    stock = models.ForeignKey(Stock,on_delete=models.CASCADE)
    open = models.DecimalField(decimal_places=4, max_digits=4)
    high = models.DecimalField(decimal_places=4, max_digits=4)
    low = models.DecimalField(decimal_places=4, max_digits=4)
    close = models.DecimalField(decimal_places=4, max_digits=4)
    volume = models.IntegerField()
    api_date = models.DateField()


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



class IndicatorCalculationData(BaseModel):
    indicators = [('STOCHASTIC','STOCHASTIC'),('RSI',"RSI"),
                  ('MACD','MACD'),('ADR','ADR'),]
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    indicator = models.CharField(choices=indicators,max_length=100)
    operation_data = models.TextField(default="{}")


    def __str__(self):
        return f"<{self.symbol}>"


