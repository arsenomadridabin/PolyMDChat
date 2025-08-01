from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Conversation(models.Model):
    """A conversation between a user and the AI"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Conversation {self.id} - {self.title or 'Untitled'}"
    
    class Meta:
        ordering = ['-updated_at']

class Message(models.Model):
    """Individual messages in a conversation"""
    MESSAGE_TYPES = [
        ('user', 'User Message'),
        ('ai', 'AI Response'),
    ]
    
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."
    
    class Meta:
        ordering = ['timestamp']

class AIModelConfig(models.Model):
    """Configuration for AI model settings"""
    name = models.CharField(max_length=100, unique=True)
    api_key = models.CharField(max_length=255, blank=True)  # Store securely in production
    model_name = models.CharField(max_length=100, default='gpt-3.5-turbo')
    max_tokens = models.IntegerField(default=1000)
    temperature = models.FloatField(default=0.7)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} ({self.model_name})"
