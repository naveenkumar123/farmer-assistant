
"""
@author: Naveen N G
@date: 30-10-2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
import ollama

class LlamaChat:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        load_dotenv(override=True)

    def chat(self, messages, stream: bool = False, response_format: str = ''):
        response = ollama.chat(
            model=self.model,
            messages=messages,
            stream=stream,
            format=response_format
        )
        if stream:
            return response
        else:
            return response['message'].content
    
    def chat_result(self, messages):
        response = ollama.chat(
            model=self.model,
            messages=messages
        )
        return response
    
    def chat_with_tool(self, messages: list, tools: list = None):
        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools
        )
        return response