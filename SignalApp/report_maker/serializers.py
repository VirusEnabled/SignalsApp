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
                  'k',

                  # 'bullet','f_stoch','f_rsi','f_macd',
                  # 'rsi','adr','macd','signal',
                  ]
        depth = 1


class HistoricalDataTransactionDetailSerializer(ModelSerializer):
    class Meta:
        model = HistoricalTransactionDetail
        fields = [
            'historical_data',
            'id_market',
            'status',
            'stop_loss_price',
            'take_profit_price',
            'avg_price',
            'entry_type',
            'number_of_entry',
            'entry_price',
            'closing_price',
            'transaction_id',
            'earning_losing_value',
            'created_at'
        ]
        depth = 1