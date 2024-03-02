from django.urls import path
from .views import CustomChatGPT

urlpatterns = [
    path('chat/', CustomChatGPT, name='chat'),
]
