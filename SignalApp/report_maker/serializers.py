from rest_framework.serializers import ModelSerializer
from .models import *

class StockSerializer(ModelSerializer):
    class Meta:
        model = Stock
        fields = ['id', 'symbol', 'stock_details','prty',
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
            'number_of_unities',
            'entry_price',
            'closing_price',
            'transaction_id',
            'earning_losing_value',
            'created_at'
        ]
        depth = 1


class ConsolidatedTransactionsSerializer(ModelSerializer):
    class Meta:
        model = ConsolidatedTransactions
        fields = [
        'stock_id',
        'net_profit',
        'gross_profit',
        'gross_loss',
        'profit_factor',
        'total_closed_trades',
        'total_open_trades',
        'number_winning_trades',
        'number_losing_trades',
        'percent_profitable',
        'average_trade',
        'average_winning_trade',
        'average_losing_trades',
        'ratio_avg_win_avg_loss',
        'largest_winning_trades',
        'largest_losing_trade',
        'avg_bars_winning',
        'avg_bars_losing'
        ]
        depth = 1