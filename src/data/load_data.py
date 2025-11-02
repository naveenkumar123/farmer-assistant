
"""
@author: Naveen N G
@date: 28-10-2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

class DataLoader:

    def __init__(self):
        pass

    def get_saved_data(self):
        with open(f'{current_dir}/train.pkl', 'rb') as file:
            train = pickle.load(file)

        with open(f'{current_dir}/test.pkl', 'rb') as file:
            test = pickle.load(file)
        
        return [train, test]