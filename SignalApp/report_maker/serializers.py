from rest_framework.serializers import ModelSerializer
from .models import *

class StockSerializer(ModelSerializer):
    class Meta:
        model = Stock
        fields = ['symbol', 'stock_details','prty',
                  'created_at'
                  ]
        depth = 1


class HistoricalDataSerializer(ModelSerializer):
    class Meta:
        model = HistoricalData
        fields = ['open','high','low',
                  'close', 'volume', 'datetime',
                  'k','bullet','f_stoch','f_rsi','f_macd',
                  'rsi','adr','macd','signal','historicaltransactiondetail'
                  ]
        depth = 1

