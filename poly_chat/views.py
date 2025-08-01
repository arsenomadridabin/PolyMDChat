from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.contrib import messages
from django.db import models
import json

from .models import Conversation, Message, AIModelConfig
from .ai_service import get_ai_service

def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('poly_chat:conversation_list')
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form})

@login_required
def ai_settings(request):
    """Manage AI configuration settings"""
    if request.method == 'POST':
        try:
            # Get or create AI config
            config, created = AIModelConfig.objects.get_or_create(
                is_active=True,
                defaults={
                    'name': 'DeepSeek Configuration',
                    'model_name': 'deepseek-chat',
                    'max_tokens': 1000,
                    'temperature': 0.7,
                    'is_active': True
                }
            )
            
            # Update configuration
            config.model_name = request.POST.get('model_name', 'checkpoint-2200')
            config.max_tokens = int(request.POST.get('max_tokens', 1000))
            config.temperature = float(request.POST.get('temperature', 0.7))
            config.save()
            
            messages.success(request, 'AI configuration updated successfully!')
            return redirect('poly_chat:ai_settings')
            
        except Exception as e:
            messages.error(request, f'Error updating configuration: {str(e)}')
    
    # Get current configuration
    config = AIModelConfig.objects.filter(is_active=True).first()
    if not config:
        config = AIModelConfig.objects.create(
            name='DeepSeek Configuration',
            model_name='deepseek-chat',
            max_tokens=1000,
            temperature=0.7,
            is_active=True
        )
    
    return render(request, 'poly_chat/ai_settings.html', {'config': config})

@login_required
def conversation_list(request):
    """Display list of user's conversations"""
    search_query = request.GET.get('search', '').strip()
    conversations = Conversation.objects.filter(user=request.user)
    
    if search_query:
        conversations = conversations.filter(
            models.Q(title__icontains=search_query) |
            models.Q(messages__content__icontains=search_query)
        ).distinct()
    
    return render(request, 'poly_chat/conversation_list.html', {
        'conversations': conversations,
        'search_query': search_query
    })

@login_required
def chat_view(request, conversation_id=None):
    """Main chat interface"""
    if conversation_id:
        conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
    else:
        # Create new conversation
        conversation = Conversation.objects.create(user=request.user)
        return redirect('poly_chat:chat_view', conversation_id=conversation.id)
    
    messages = conversation.messages.all().order_by('timestamp')
    user_conversations = Conversation.objects.filter(user=request.user).order_by('-updated_at')
    
    return render(request, 'poly_chat/chat.html', {
        'conversation': conversation,
        'messages': messages,
        'user_conversations': user_conversations
    })

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def send_message(request, conversation_id):
    """API endpoint to send a message and get AI response"""
    try:
        conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        # Save user message
        user_msg = Message.objects.create(
            conversation=conversation,
            content=user_message,
            message_type='user'
        )
        
        # Generate AI response
        ai_service = get_ai_service()
        ai_response = ai_service.generate_response(conversation, user_message)
        
        # Save AI response
        ai_msg = Message.objects.create(
            conversation=conversation,
            content=ai_response,
            message_type='ai'
        )
        
        # Update conversation title if it's the first message
        if conversation.messages.count() == 2:  # user message + ai response
            conversation.title = user_message[:50] + "..." if len(user_message) > 50 else user_message
            conversation.save()
        
        return JsonResponse({
            'success': True,
            'user_message': {
                'id': user_msg.id,
                'content': user_msg.content,
                'timestamp': user_msg.timestamp.isoformat()
            },
            'ai_response': {
                'id': ai_msg.id,
                'content': ai_msg.content,
                'timestamp': ai_msg.timestamp.isoformat()
            }
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def get_messages(request, conversation_id):
    """API endpoint to get messages for a conversation"""
    try:
        conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        messages = conversation.messages.all().order_by('timestamp')
        
        message_list = []
        for message in messages:
            message_list.append({
                'id': message.id,
                'content': message.content,
                'message_type': message.message_type,
                'timestamp': message.timestamp.isoformat()
            })
        
        return JsonResponse({'messages': message_list})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def delete_conversation(request, conversation_id):
    """Delete a conversation"""
    conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
    conversation.delete()
    return redirect('poly_chat:conversation_list')
