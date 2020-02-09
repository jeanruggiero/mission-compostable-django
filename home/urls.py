from django.urls import path
from . import views

urlpatterns = [
    path('', views.simple_upload, name="home"),
    path('results', views.results, name="results")
]