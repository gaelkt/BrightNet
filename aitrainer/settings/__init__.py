# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:29:12 2021

@author: gaelk
"""
from __future__ import absolute_import, unicode_literals
import sys

sys.path.append('./../')
sys.path.append('..')



# This will make sure the app is always imported when
# Django starts so that shared_task will use this app.
# from celery import app as celery_app



__all__ = ('celery_app',)