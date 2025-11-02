
"""
@author: Naveen N G
@date: 28-10-2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from data_retriever import DataRetriever




class DatasetLoader:


    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
    MIN_TOKENS = 150 # Any less than this, and we don't have enough useful content
    MAX_TOKENS = 160 # Truncate after this many tokens. Then after adding in prompt text, we will get to around 180 tokens
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Answer is:"

    REMOVALS = ["Farmer asked query on Weather", "Asking about?", "information about", "Information about", 
                "Farmer asked", "Ask about information on", "Information regarding", 
                "ASKING ABOUT", "FARMER ASKED QUERY ON", "???????? ??? ???? ??????"]


    def __init__(self, year = 2025):
        self.year = year



    def load(self, max_page=50, workers=10):
        """
        Fetch pages asynchronously using threads, stop when a page returns no data.
        """
        all_results = []
        stop = False
        page = 1

        data_retriever = DataRetriever()

        while page <= max_page and not stop:
            pages = list(range(page, min(page + workers, max_page + 1)))
            print(f"Fetching pages {pages}...")

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_page = {executor.submit(data_retriever.fetch_kcc_data, self.year, p): p for p in pages}

                results = []
                for future in as_completed(future_to_page):
                    p, data = future.result()
                    results.append((p, data))

            # Sort results by page number so they're in order
            # results.sort(key=lambda x: x[0])

            # Process results and check for stop condition
            for p, data in results:
                if not data or len(data) == 0:
                    print(f"Page {p} returned no data. Stopping further requests.")
                    stop = True
                    break
                all_results.extend(data)

            page += workers

        print(f"Done! Total records fetched: {len(all_results)}")
        return all_results
    
    def load_data(self, file_path):
        """
        Reads a CSV file and returns its content as a list of dictionaries.
        Each dictionary represents a row, with column headers as keys.
        """
        data_list = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
            dict_reader = csv.DictReader(csv_file)
            for row in dict_reader:
                data = {}
                data['StateName'] = row.get('StateName', '').strip()
                data['Sector'] = row.get('Sector', '').strip()
                data['Category'] = row.get('Category', '').strip()
                data['Crop'] = row.get('Crop', '').strip()
                data['QueryType'] = row.get('QueryType', '').strip()
                data['Answer'] = row.get('KccAns', '').strip()

                query = self.filter_details(row.get('QueryText', ''))
                if len(query) < 1 or data['QueryType'] == "Weather":
                    continue
                
                data['Query'] = query.strip()
                prompt = self.get_prompt(data)
                test_prompt = self.test_prompt(prompt)
                data['prompt'] = prompt
                data['test_prompt'] = test_prompt

                data_list.append(data)

        return data_list

    def filter_details(self, query):
        """
        Clean the details field by removing unwanted substrings.
        """
        for remove in self.REMOVALS:
            query = query.replace(remove, "")
        return query



    def get_prompt(self, data):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        """
        prompt = f"""In the {data.get('Sector')} sector, under {data.get('Category')} category. Could you provide infromation about/on {data.get('Query')}.\n
                        This question is related to {data.get('QueryType')}.  \n\n\n"""
        prompt += f"{self.PREFIX}{data.get('Answer')}\n"
        # token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))
        return prompt

    def test_prompt(self, prompt):
        """
        Return a prompt suitable for testing, with the actual price removed
        """
        return prompt.split(self.PREFIX)[0] + self.PREFIX


