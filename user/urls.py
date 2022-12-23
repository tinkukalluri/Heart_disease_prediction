from django.contrib import admin
from django.urls import path , include
from . import views



urlpatterns = [
    path('login' , views.login_view),
    path('authenticate' , views.authenticate_view),
    path('logout' , views.logout),
]