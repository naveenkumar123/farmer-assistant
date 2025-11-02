
"""
@author: Naveen N G
@date: 28-10-2025
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
from dotenv import load_dotenv
import csv

load_dotenv()

class DataRetriever:
    def __init__(self):
        self.data_url = os.getenv('KCC_DATA_URL')

    def fetch_kcc_data(self, year, page_no):
        try:
            response = requests.get(self.data_url, params={"year": year,"page": page_no})
            response.raise_for_status()  # Raise an error for bad status codes
            print(f"Response received")
            result = response.json()
            return page_no, result.get('Data')
        except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching the KCC data: {e}")
                return None
        

def dataPuller():
    data_retriever = DataRetriever()
    kcc_data = data_retriever.fetch_kcc_data()
    data_retriever.save_data_to_csv(kcc_data)


if __name__ == "__main__":
    data_retriever = DataRetriever()
    kcc_data = data_retriever.fetch_kcc_data(2025, 1)
    print(kcc_data)

# Use this command to run : python src/data/data_retriever.py 