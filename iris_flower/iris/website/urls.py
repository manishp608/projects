from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_iris, name='predict_iris')
]
