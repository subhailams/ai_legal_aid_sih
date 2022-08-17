# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.home import views

from django.conf import settings #add this
from django.conf.urls.static import static


urlpatterns = [

    # The home page
    path('', views.index, name='home'),
    path('adddb', views.addCasetoDB, name='adddb'),
    path('sec', views.sec, name='sec'),
    path('analysis', views.case_analysis, name='analysis'),
    path('trans', views.translate, name='trans'),
    path('uploaded_cases', views.uploaded_cases, name='uploaded_cases'),
    re_path('similar/(?P<id>[\w-]+)/$', views.get_similar_cases, name='similar'),
   
    #path('analysis/',views.analysis),
    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),
    
] 
