import os
from typing import Optional

def setup_langchain():
    """Initialize LangChain with proper error handling"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.memory import ConversationBufferWindowMemory
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        return {
            'ChatOpenAI': ChatOpenAI,
            'ConversationBufferWindowMemory': ConversationBufferWindowMemory,
            'LLMChain': LLMChain,
            'PromptTemplate': PromptTemplate,
            'available': True,
            'version': 'new'
        }
    except ImportError:
        try:
            # Fallback for older versions
            from langchain.chat_models import ChatOpenAI
            from langchain.memory import ConversationBufferWindowMemory
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            
            return {
                'ChatOpenAI': ChatOpenAI,
                'ConversationBufferWindowMemory': ConversationBufferWindowMemory,
                'LLMChain': LLMChain,
                'PromptTemplate': PromptTemplate,
                'available': True,
                'version': 'old'
            }
        except ImportError:
            return {'available': False}