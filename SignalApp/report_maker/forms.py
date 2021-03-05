from django.forms import Form, fields, widgets, ModelForm, ModelChoiceField, ChoiceField
from .models import *
import re
import requests as r
from django.db.models.query import QuerySet
from django.forms.fields import ChoiceField
from .TradierDataFetcherService import TradierDataHandler

class LoginForm(Form):
    email = fields.EmailField()
    password = fields.CharField()

    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.fields['email'].widget = widgets.TextInput(attrs={'class': 'page-login-field top-15  form-control',
                                                                  'placeholder': 'Email Address'})
        self.fields['email'].required = False
        self.fields['email'].validators = []
        self.fields['password'].required = False
        self.fields['password'].widget = widgets.PasswordInput(
            attrs={'class': 'page-login-field bottom-20 form-control','placeholder': 'Password'})



class StockChoiceField(ChoiceField):

    def __init__(self):
        choices = [(f'{market["Code"]}', f'{market["Code"]}'.upper())
                   for market in TradierDataHandler.load_markets()[:5000]]
        super().__init__(choices=choices)


class StockGenericOperation(Form):
    stock = StockChoiceField()
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for field in self.fields.keys():
            self.fields[field].widget.attrs['class'] = 'form-control'

        self.fields['stock'].widget.attrs['class']='form-control front-selector'



class HistoricalStockOperationForm(Form):
    stock = StockChoiceField()
    start_date = fields.DateField()
    end_date = fields.DateField()
    interval=ChoiceField(choices=[('Daily','daily'),
                                  ('Weekly','weekly'),
                                  ('Monthly','monthly')])

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for field in self.fields.keys():
            self.fields[field].widget.attrs['class']='form-control'
        self.fields['start_date'].widget = widgets.DateInput({'class':'form-control', 'type': 'date'})
        self.fields['end_date'].widget = widgets.DateInput({'class':'form-control', 'type': 'date'})



class StockOperationForm(Form):
    shift_options = [('tick','tick'),
                                  ('1min','1min'),
                                  ('5min','5min'),
                                  ('15min', '15min'),
                                  ]
    stock = StockChoiceField()
    start_date = fields.DateField()
    start_time = fields.TimeField()
    end_date = fields.DateField()
    end_time = fields.TimeField()
    interval=ChoiceField(choices=shift_options)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for field in self.fields.keys():
            self.fields[field].widget.attrs['class']='form-control'
        self.fields['start_date'].widget = widgets.DateInput({'class':'form-control', 'type': 'date'})
        self.fields['end_date'].widget = widgets.DateInput({'class':'form-control', 'type': 'date'})
        self.fields['start_time'].widget = widgets.DateInput({'class': 'form-control', 'type': 'time'})
        self.fields['end_time'].widget = widgets.DateInput({'class': 'form-control', 'type': 'time'})

