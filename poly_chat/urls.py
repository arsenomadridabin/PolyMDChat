from django.urls import path
from . import views

app_name = 'poly_chat'

urlpatterns = [
    path('register/', views.register, name='register'),
    path('ai-settings/', views.ai_settings, name='ai_settings'),
    path('', views.conversation_list, name='conversation_list'),
    path('chat/', views.chat_view, name='chat_view'),
    path('chat/<int:conversation_id>/', views.chat_view, name='chat_view'),
    path('api/chat/<int:conversation_id>/send/', views.send_message, name='send_message'),
    path('api/chat/<int:conversation_id>/messages/', views.get_messages, name='get_messages'),
    path('chat/<int:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
] 