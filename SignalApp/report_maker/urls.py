from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static


handler404 = 'report_maker.views.handler404'
handler500 = 'report_maker.views.handler500'
app_name='report_maker'

urlpatterns = [
    path('',LoginView.as_view(),name='login'),
    path('dashboard', Dashboard.as_view(), name='dashboard'),
    path('logout', logout_user, name='logout'),

    ]+static(settings.STATIC_URL, document_root=settings.MEDIA_URL)