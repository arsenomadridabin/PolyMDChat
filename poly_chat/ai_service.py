import requests
import json
import torch
import logging
from typing import Dict, List, Optional
from .models import AIModelConfig, Message, Conversation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
current_model_path = None

def load_model(model_path=None):
    """Load the fine-tuned Mixtral model"""
    global model, tokenizer, current_model_path
    
    try:
        # Use provided model path or default
        if model_path is None:
            model_path = "checkpoint-2200"  # Default path
        
        # Check if model is already loaded with the same path
        if model is not None and tokenizer is not None and current_model_path == model_path:
            logger.info(f"Model already loaded from: {model_path}")
            return
        
        logger.info(f"Loading model from: {model_path}")
        
        # Import here to avoid loading on startup
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        current_model_path = model_path
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def is_model_loaded():
    """Check if model is currently loaded"""
    global model, tokenizer, current_model_path
    return model is not None and tokenizer is not None and current_model_path is not None

def get_current_model_path():
    """Get the path of the currently loaded model"""
    global current_model_path
    return current_model_path

def reload_model(model_path=None):
    """Reload the model with a new path"""
    global model, tokenizer, current_model_path
    
    # Clear existing model and tokenizer
    model = None
    tokenizer = None
    current_model_path = None
    
    # Load new model
    load_model(model_path)

def format_chat_history(messages):
    """Format messages for multi-turn chat using the proven chat_cli.py approach"""
    prompt = ""
    for i, msg in enumerate(messages):
        if msg["role"] == "system" and i == 0:
            prompt = f"<s>[INST] <<SYS>>\n{msg['content']}\n<</SYS>>\n\n"
        elif msg["role"] == "user":
            prompt += f"<s>[INST] {msg['content']} [/INST]"
        elif msg["role"] == "assistant":
            prompt += f" {msg['content']} </s>"
    return prompt

class AIService:
    """Service for handling AI model interactions"""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.api_key = config.api_key
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
    
    def generate_response(self, conversation: Conversation, user_message: str) -> str:
        """
        Generate AI response based on conversation history and user message
        """
        # Get conversation history
        messages = conversation.messages.all().order_by('timestamp')
        
        # Prepare conversation context for AI
        conversation_history = []
        
        # Add system message (using the same system prompt from chat_cli.py)
        conversation_history.append({
            "role": "system",
            "content": "You are a helpful assistant specialized in polymers and molecular simulations."
        })
        
        # Add conversation history
        for message in messages:
            role = "user" if message.message_type == "user" else "assistant"
            conversation_history.append({
                "role": role,
                "content": message.content
            })
        
        # Add current user message
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Call appropriate model based on configuration
        if self.config.model_type == 'local':
            return self._call_local_model(conversation_history)
        elif self.config.model_type == 'deepseek':
            return self._call_deepseek_api(conversation_history)
        elif self.config.model_type == 'openai':
            return self._call_openai_api(conversation_history)
        elif self.config.model_type == 'anthropic':
            return self._call_anthropic_api(conversation_history)
        else:
            # Default to local model
            return self._call_local_model(conversation_history)
    
    def _call_local_model(self, messages: List[Dict]) -> str:
        """
        Call local fine-tuned Mixtral model directly using proven chat_cli.py approach
        """
        global model, tokenizer, current_model_path
        
        try:
            # Check if we need to load or reload the model
            model_path = self.model_name
            
            logger.info(f"Current model path: {current_model_path}, Requested path: {model_path}")
            logger.info(f"Model loaded: {model is not None}, Tokenizer loaded: {tokenizer is not None}")
            
            # Load model if not already loaded or if model path has changed
            if (model is None or tokenizer is None or 
                current_model_path != model_path):
                
                logger.info(f"Loading model from: {model_path}")
                load_model(model_path)
            else:
                logger.info("Using already loaded model")
            
            # Format messages for the model using proven approach
            prompt = format_chat_history(messages)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response using same parameters as chat_cli.py
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response using same approach as chat_cli.py
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reply = output_text.split("[/INST]")[-1].strip().split("</s>")[0].strip()
            
            return reply
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def _call_deepseek_api(self, messages: List[Dict]) -> str:
        """
        Call DeepSeek API (kept for reference)
        """
        try:
            # DeepSeek API endpoint
            url = "https://api.deepseek.com/v1/chat/completions"
            
            # Headers for DeepSeek API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Request payload
            payload = {
                "model": self.model_name or "deepseek-chat",
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            # Make API request
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                error_msg = f"DeepSeek API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                except:
                    pass
                return f"Error: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Error: Network issue - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _call_openai_api(self, messages: List[Dict]) -> str:
        """
        Call OpenAI API (kept for reference)
        """
        # Implementation for OpenAI - kept for future use
        pass
    
    def _call_anthropic_api(self, messages: List[Dict]) -> str:
        """
        Call Anthropic API (Claude) - kept for reference
        """
        # Implementation for Anthropic Claude
        pass

def get_ai_service() -> AIService:
    """
    Get the active AI service configuration
    """
    try:
        config = AIModelConfig.objects.filter(is_active=True).first()
        if not config:
            # Create a default config for local model
            config = AIModelConfig.objects.create(
                name="Local Mixtral Model",
                model_name="checkpoint-2200",
                max_tokens=1000,
                temperature=0.7,
                is_active=True
            )
        return AIService(config)
    except Exception as e:
        raise Exception(f"Failed to initialize AI service: {str(e)}") 