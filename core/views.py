from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FC
import numpy as np
import json 
import datetime as dt
from io import BytesIO
import base64
from . import machine_backend
from .models import Dividend_Data
import mpld3
# Create your views here.

def index(request):
    start_date = dt.datetime(2016, 1, 1)
    end_date = dt.datetime(2021, 1, 1 )
    yesterday_date = dt.datetime(2024, 2, 11)
    print(yesterday_date)
    if request.method == "POST":
        symbol = request.POST.get('symbol')


        data = machine_backend.get_stock_data(symbol, start_date, end_date)
        generate_graph_data = machine_backend.generate_data(symbol, data, yesterday_date)

        graph = generate_graph_data['graph']
        predicted_share_price = (generate_graph_data['prediction'][0][0])
        percentage = (generate_graph_data['percentage'][0][0])
        

        html_content = mpld3.fig_to_html(graph)
        print(f'predicted share price: {predicted_share_price}, percentage: {percentage}')


        return JsonResponse({'message': html_content, 'prediction': predicted_share_price, 'symbol': symbol, 'percentage': percentage})

    symbols = Dividend_Data.objects.all()


    context = {
        "symbols"    : symbols,
        
    }
    return render(request, 'core/index.html', context)

def save_dividend_data(request):
    symbols = machine_backend.get_sp500_members('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    print(symbols)
    for x in range(len(symbols['symbols'])):

        new_dividend = Dividend_Data(
            symbol  = symbols['symbols'][x],
            company = symbols['companies'][x],
            sector  = symbols['sectors'][x]
        )
        # Save the new dividend data
        new_dividend.save()

    return HttpResponse("Data has been saved to the database.")