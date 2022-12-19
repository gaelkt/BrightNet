# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:29:12 2021

@author: gaelk
"""

from .base import *

# Added Host for qualif
ALLOWED_HOSTS = [
                 '.qualif.ams2.cloud.weemo.com',
                 '127.0.0.1']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'aitrainerdb',
        'USER': 'aitraineruser@aitrainerdb01',
        'PASSWORD': 'aiSoh4Yeipiej(ie',
        'HOST': 'aitrainerdb1.qualif.ams2.cloud.weemo.com',  # Or an IP Address that your database is hosted on
        'PORT': '3306',
        }

}

SECRET_KEY = 'django-insecure-=tot_2ooe)cpgw)!nj-_o4=a57(g8j0x6sod+o*$q((3ybg%r9'


CENTRAL_CONN_STR_AZURE = 'DefaultEndpointsProtocol=https;AccountName=sightcallmicroservices;AccountKey=AXHnvjsnb+3ZXmNBwCsK7pE7P2e7kgiHr7zkxO6ebETooVRZBHgWdk1fSLfE5BH/0/YSxY2ucJIZdcKhrYnlrg==;EndpointSuffix=core.windows.net'
CENTRAL_AZURE_CONTAINER = 'aitrainermodels-qualif'

AUTHORIZATION = 'Token SightCall28072021Aitrainer'

DEFAULT_WEBHOOK_URL='https://8960-91-173-26-213.ngrok.io'

# path for the file log
FILE_LOG = r''

# Directory for local datasets 
LOCAL_DIRECTORY_DATASET = "to_fill_if_needed"



