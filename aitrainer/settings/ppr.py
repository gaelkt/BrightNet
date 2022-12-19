# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:29:12 2021

@author: gaelk
"""

# Settings for PPR
from .base import *

# Added Host for qualif
ALLOWED_HOSTS = [
                 '.ppr.was10.cloud.weemo.com',
                 '127.0.0.1']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'aitrainerdb',
        'USER': 'aitraineruser@aitrainerdbppr01',
        'PASSWORD': 'Lcao79Yeiptej,41',
        'HOST': 'aitrainerdb1.ppr.was10.cloud.weemo.com',  # Or an IP Address that your database is hosted on
        'PORT': '3306',
        }

}

SECRET_KEY = 'django-insecure-=tot_2ooe)cpgw)!nj-_o4=a57(g8j0x6sod+o*$q((3ybg%r9'


CENTRAL_CONN_STR_AZURE = 'DefaultEndpointsProtocol=https;AccountName=sightcallmicroservices;AccountKey=AXHnvjsnb+3ZXmNBwCsK7pE7P2e7kgiHr7zkxO6ebETooVRZBHgWdk1fSLfE5BH/0/YSxY2ucJIZdcKhrYnlrg==;EndpointSuffix=core.windows.net'
CENTRAL_AZURE_CONTAINER = 'aitrainermodels-ppr'

AUTHORIZATION = 'Token sAJvDWa5xubR2f3fmTxxH2nFGwrey3gFtRwr7pzz'

DEFAULT_WEBHOOK_URL='https://2fe4-2a01-e0a-346-b560-4d89-6a02-d68a-d9a9.ngrok.io'

# path for the file log
FILE_LOG = r''

# Directory for local datasets 
LOCAL_DIRECTORY_DATASET = "to_fill_if_need"

