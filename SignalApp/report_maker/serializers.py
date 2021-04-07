from rest_framework.serializers import ModelSerializer
from .models import *

class StockSerializer(ModelSerializer):
    class Meta:
        model = Stock
        fields = ['symbol', 'created_at']
        depth = 1


class HistoricalDataSerializer(ModelSerializer):
    class Meta:
        model = HistoricalData
        fields = ['stock', 'open','high','low',
                  'close','volume','api_date']