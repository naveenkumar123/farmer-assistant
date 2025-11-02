"""
@author: Naveen N G
@date: 27-10-2025
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from huggingface_hub import login


load_dotenv()


def login_hf():
    HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
    login(HUGGING_FACE_TOKEN)