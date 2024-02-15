from django.db import models

# Create your models here.

class Dividend_Data (models.Model):
    symbol = models.CharField(max_length=100)
    company = models.CharField(max_length=155)
    sector = models.CharField(max_length=200)

    