
"""
@author: Naveen N G
@date: 30-10-2025
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import math
from tqdm import tqdm
from dotenv import load_dotenv
from hg_login import login_hf
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
from peft import PeftModel
import matplotlib.pyplot as plt

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = "naveenng10/farming-assistant"
DATASET_NAME = 'naveenng10/former-assistant-query-data'

QUANT_4_BIT = True

# Used for writing to output in color
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}


load_dotenv()
login_hf()


class EvaluateFinetuneModel:
    def __init__(self):
        self.load_model()

    def load_data(self):
        dataset = load_dataset(DATASET_NAME)
        train = dataset['train']
        test = dataset['test']
        return train, test

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

    def model_predict(self, prompt):
        set_seed(42)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=150, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0])
        return self.extract_answer(response)
    
    def extract_answer(self, response):
        if "Answer is:" in response:
            contents = response.split("Answer is:")[1]
            return contents.strip()
        return ""
    

    ##############
    # Goole colab notebook link : https://colab.research.google.com/drive/1E_ncDcmI9JdeuxLHZCVVrQleO9CUVHEW#scrollTo=t480-w_1kv_o
    ###############