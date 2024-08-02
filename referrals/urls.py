from django.urls import path
from . import views
from .views import add_user, get_users, predict_interest

urlpatterns = [
    path('add_user/', add_user, name='add_user'),
    path('get_users/', get_users, name='get_users'),
    path('predict_interest/', views.predict_interest, name='predict_interest'),
    path('predict_interest_form/', views.predict_interest_form, name='predict_interest_form'),
    path('predict_interest_result/', views.predict_interest_result, name='predict_interest_result')
]
