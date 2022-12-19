"""aitrainer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""


import sys
sys.path.insert(0, "../")
from django.contrib import admin
from django.urls import path
from trainer import views


urlpatterns = [
    path('admin/', admin.site.urls),

    path('starttraining/', views.form_start_training),

    path('checkserver/', views.check_server),

    # path('getmodel/<str:profile_id>/', views.form_get_model),
    # path('getmodel/<str:profile_id>/<str:project_name>/', views.form_get_model),
    # path('getmodel/', views.form_get_model),  
    
    path('statustraining/<str:profile_id>/', views.form_status_training),
    path('statustraining/<str:profile_id>/<str:project_name>/', views.form_status_training),
    path('statustraining/', views.form_status_training),
    
    
]
