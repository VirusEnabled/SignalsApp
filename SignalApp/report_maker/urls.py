from django.urls import path,include
from .views import *
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.routers import SimpleRouter

api_router = SimpleRouter()
api_router.register('stock_handler',StockViewSet,'stock_handler')
handler404 = 'report_maker.views.handler404'
handler500 = 'report_maker.views.handler500'
app_name='report_maker'

urlpatterns = [
    path('',LoginView.as_view(),name='login'),
    path('dashboard', Dashboard.as_view(), name='dashboard'),
    path('logout', logout_user, name='logout'),
    path('generate_graphs', generate_graphs, name='generate_graphs'),
    path('api/',include((api_router.urls,'api_stocks')),name='api_stocks'),
    path('api/stock_handler/list', stock_list, name='stock_list'),
    path('api/stock_handler/get_last_entry_record', get_last_entry_record, name='get_last_entry_record'),
    path('api/stock_handler/get_entry_detail', get_entry_detail, name='get_entry_detail')
    ]+static(settings.STATIC_URL, document_root=settings.MEDIA_URL)