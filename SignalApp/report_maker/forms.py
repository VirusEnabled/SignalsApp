from django.forms import Form, fields, widgets, ModelForm, ModelChoiceField
from .models import *
import re
import requests as r
from django.db.models.query import QuerySet
from django.forms.fields import ChoiceField


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
