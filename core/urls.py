from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('get-symbols/', views.save_dividend_data,name='save-dividend-data')
]