"""
@author: Naveen N G
@date: 29-10-2025
@description: Upload the trained and test prompt dataset to Huggingface.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from hg_login import login_hf
import random
import pickle

load_dotenv()
login_hf()

current_dir = os.path.dirname(os.path.abspath(__file__))


class UploadDataset:

    def __init__(self, items):
        self.sample = items

    def upload(self):
        random.seed(42)
        random.shuffle(self.sample)
        train = self.sample
        test = self.sample[10_000:50_000] # 30,000 for testing
        print(f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items")



        train_prompts = [item['prompt'] for item in train]
        train_ans = [item['Answer'] for item in train]
        test_prompts = [item['test_prompt'] for item in test]
        test_ans = [item['Answer'] for item in test]


        # Create a Dataset from the lists
        train_dataset = Dataset.from_dict({"query": train_prompts, "answer": train_ans})
        test_dataset = Dataset.from_dict({"query": test_prompts, "answer": test_ans})
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        DATASET_NAME = "naveenng10/former-assistant-query-data"
        dataset.push_to_hub(DATASET_NAME, private=True)

        # Let's pickle the training and test dataset so we don't have to execute all this code next time!
        with open(f'{current_dir}/train.pkl', 'wb') as file:
            pickle.dump(train, file)

        with open(f'{current_dir}/test.pkl', 'wb') as file:
            pickle.dump(test, file)