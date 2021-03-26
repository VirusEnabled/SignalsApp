import sys
import os
from django.shortcuts import render, redirect, reverse, get_object_or_404,HttpResponseRedirect, HttpResponse
from django.views.generic import TemplateView, DetailView, ListView, RedirectView, GenericViewError
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin,AccessMixin
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.views import View
from django.template.loader import render_to_string
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.models import AnonymousUser
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout, authenticate, user_logged_in, user_logged_out
from django.contrib import messages
from rest_framework.parsers import JSONParser
from django.core.files import File as DF
from functools import wraps
from django.http import JsonResponse, QueryDict as qd
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from datetime import timedelta, datetime
from .helpers import *
from .forms import *
from django.views.defaults import page_not_found, server_error
from django.conf import settings
from datetime import time

def handler404(request, exception, template_name=None):
    template_name = 'includes/_error_page.html'
    response = HttpResponse(request,template_name)
    response.status_code = status.HTTP_404_NOT_FOUND
    response.content = {'status': response.status_code}
    return render(request, template_name, {'status':response.status_code})
    # return page_not_found(request,exception,template_name)


def handler500(request, template_name=None):
    template_name = 'includes/_error_page.html'
    response = HttpResponse(request, template_name)
    response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    response.content = {'status': response.status_code}

    return response
    # return server_error(request,template_name)

class LoginView(View):
    template_name = 'report_maker/login.html'
    extra_context = {}
    form_class = LoginForm
    success_redirect = 'report_maker:dashboard'

    def load_form(self,request):
        """
        loads the form to be used
        :return: None
        """
        self.form = self.form_class(request.POST)
        self.next = request.GET.get('next') if 'next' in request.GET.keys() else None

    def get(self, request):
        """
		loads the template and the form for the user
		when requested with the http GET method
		:param request:  HTTP request object
		:return: HTTP response.
		"""
        if request.user.is_authenticated:
            return redirect('report_maker:dashboard')
        self.load_form(request)
        context = {
            'form': self.form
        }
        return render(request, self.template_name, context)

    def post(self, request):
        """
		validates if whether the user is worth it to pass to the site or not.
		:param request: HTTP request object
		:return: redirect
		"""
        return self.validate_form()

    def validate_form(self):
        """
		validates if the given data and tries to log the user Through amazon
		if successful then returns it to the form valid otherwise to the form invalid.
		here is where we do the two factor authentication verify the existance of the account
		here then verify the existance of the account in Amazon. To pass, needs to pass in both parts.

		Basically the code to validate in both sides is in the form
		:return: Http response
		"""
        self.load_form(self.request)
        error_msg = "The email or password incorrect, try again."
        try:
            if self.form.is_valid():
                user_data = self.form.cleaned_data
                user = User.objects.get(email__exact=user_data['email'])
                authenticated = authenticate(username=user, password=user_data['password'])
                # print('here',self.request, self.form.is_valid(), self.form.cleaned_data)
                if authenticated:
                    return self.form_valid(authenticated)
            return self.form_invalid(error_msg)
        except ObjectDoesNotExist:
            error_msg = "The email provided doesn't exist in our system, please contact the admin."
            return self.form_invalid(error_msg)

    def form_valid(self, user):
        """
        this is the result of passing all evaluations of the user account
        :return: http response
        """
        self.request.session.set_expiry(self.request.session.get_expiry_age() * 4)
        login(self.request, user)
        if self.next:
            return HttpResponseRedirect(self.next, self.request)
        return self.redirect_success()

    def form_invalid(self, error=None):
        """
		returns the errors of the given values in the form fields.
		:return: Http response
		"""
        if not error:
            for error in self.form.errors:
                messages.error(self.request, self.form.errors[error])
        else:
            messages.error(self.request, error)
        return render(self.request, self.template_name, {'form': self.form})

    def redirect_success(self):
        """
        redirects the user to the dashboard
        :return: http redirect response
        """
        messages.success(self.request, 'You\'ve been successfully logged in')
        return redirect(self.success_redirect)


@login_required(redirect_field_name='next', login_url='report_maker:login')
def logout_user(request):
    logout(request)
    messages.success(request,'You have been successfully logged out!')
    return redirect('report_maker:login')


class Dashboard(TemplateView, LoginRequiredMixin):
    template_name = "report_maker/dashboard.html"
    login_url = 'report_maker:login'
    permission_denied_message = "In order to access to this part, you should be logged in"

    http_method_names = ['get', 'post']
    extra_context = {'available_stocks': TRADIER_API_OBJ.load_markets()[:10]}

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return super().get(request, args,kwargs)
        else:
            messages.error(request,self.permission_denied_message)
            return redirect('report_maker:login')

    def get_context_data(self, **kwargs):
        # self.extra_context['graph'] = get_stock_historical_candlestick_graph()
        base_selection_form, selection_form, historical_form = self.get_forms(self.request)
        self.extra_context['filter_form'] = selection_form
        self.extra_context['form'] = base_selection_form
        self.extra_context['historical_query_form'] = historical_form

        return super().get_context_data(**kwargs)


    def get_forms(self, request):
        """
        loads the forms needed to
        finish the operations
        :return: tuple
        """
        self.form = StockGenericOperation(request.POST)
        self.filters_form =  StockOperationForm(request.POST)
        self.historical_form = HistoricalStockOperationForm(request.POST)
        return self.form, self.filters_form, self.historical_form

    # might need to fix it when we go live
    def post(self, request):
        """
        covers the post request for the dashboard
        it should get the form
        :return: graph
        """
        form, filters, historical_form =self.get_forms(request)
        context = {}
        validate = filters.is_valid()
        if form.is_valid() or validate:
            filters_data = filters.cleaned_data if filters.is_valid() else None
            data = form.cleaned_data
            current_date = datetime.today().now()
            today = current_date.strftime("%Y-%m-%d %H:%m")
            # a_year_back = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d %H:%m")
            start_year =(datetime(current_date.year,1,
                                  current_date.day)+timedelta(days=2)).strftime("%Y-%m-%d %H:%m")
            saved, value = settings.REDIS_OBJ.save_graph_refresh_time(user=request.user, symbol=data['stock'], refresh_time=today)
            if not saved:
                messages.error(request, f"There's was a problem generating the graph: "
                                        f"We couldn't save the last refresh time because: {value}")
                return self.get(request)

            if validate:
                if filters_data['start_date'] > filters_data['end_date']:
                    messages.error(request, f"There's was a problem generating the graph: "
                                            f"The dates provided don't match.\n the start date can't be greater or "
                                            f"equal then the end date, try again.")
                    return self.get(request)

            passed, graphs = generate_graph_calculations(symbol=data['stock'] if not filters_data else \
                                                                    filters_data['stock'],
                                                                   start_date=start_year if not filters_data else\
                                                                       self.process_time(filters_data['start_date'],
                                                                                         time(hour=7,minute=0,second=0)),
                                                                   end_date=today if not filters_data else \
                                                                       self.process_time(filters_data['end_date'],
                                                                                         time(hour=21, minute=0,
                                                                                              second=0)),
                                                                    interval='15min'
                                                                   )
            if passed:
                messages.success(request,"The graph was generated successfully!")
                context = self.get_context_data()
                print(graphs.keys())
                for key in graphs:
                    context[key] = graphs[key]

                return render(request, self.template_name, context=context)

            else:
                messages.error(request,
                               f"There's was a problem generating the graph: {graphs['error']}, try again.")

        else:
            messages.error(request, f"There's was a problem with your request, "
                                    f"please try again.{filters.errors if not filters.is_valid() else ''}")
        return self.get(request)


    def process_time(self, dt: date, dtime: time) -> str:
        """
        converts the two given object into one
        :param dt: date object
        :param dtime: time object
        :return: str
        """
        return f"{dt.isoformat()} {dtime.isoformat()}"

@login_required(login_url="report_maker:login",redirect_field_name='next')
@api_view(http_method_names=['POST'])
def generate_graphs(request):
    """
    gets the request for the given
    data and returns the data in a json string
    :param request: http request
    :return: json
    """
    s = status.HTTP_404_NOT_FOUND
    response = {'message':''}
    data = request.data
    data['user'] = request.user.username
    # print(data)
    candlestick_template = "includes/_candlestick.html"
    passed, graph = get_stock_historical_candlestick_graph(symbol=data['selected_stock'],
                                                           start_date=data['start_date'],
                                                           end_date=data['end_date'],
                                                           interval=data['selected_interval'].lower())
    if passed:
        s = status.HTTP_200_OK
        response['message'] = "the graphs has been generated successfully"
        response['candlestick_graph'] = render_to_string(candlestick_template, {'graph':graph})
        response['candlestick_graph'] = graph

    else:
        response['message'] = graph['error']
        s = status.HTTP_400_BAD_REQUEST

    return Response(status=s, data=response)


"""
@login_required()
# @api_view(http_method_names=['POST'])
def generate_graphs(request):

    # gets the request for the given
    # data and returns the data in a json string
    # :param request: http request
    # :return: http response

    s = status.HTTP_404_NOT_FOUND
    template_name = "report_maker/dashboard.html"
    response = {'message':''}
    data = request.POST
    # data['user'] = request.user.username
    print(data)
    if request.method == 'POST':
        candlestick_template = "includes/_candlestick.html"
        passed, graph = get_stock_historical_candlestick_graph(symbol=data['selected_stock'],
                                                               start_date=data['start_date'],
                                                               end_date=data['end_date'],
                                                               interval=data['selected_interval'].lower())
        if passed:
            # s = status.HTTP_200_OK
            # response['message'] = "the graphs has been generated successfully"
            # response['candlestick_graph'] = render_to_string(candlestick_template, {'graph':graph})
            messages.success(request,"the graphs has been generated successfully")
            return render(request, template_name, {'graph':graph})
        else:
            response['message'] = graph['error']
            s = status.HTTP_400_BAD_REQUEST
            messages.error(request, graph['error'])
    return render(request, template_name, {})
        # return Response(status=s, data=response)

"""