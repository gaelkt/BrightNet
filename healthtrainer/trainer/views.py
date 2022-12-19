from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse


def checkserver(request):
    return HttpResponse("Hello, world. You're at the polls index.")
