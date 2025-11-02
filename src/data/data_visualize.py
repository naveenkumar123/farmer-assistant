
"""
@author: Naveen N G
@date: 28-10-2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from collections import Counter

load_dotenv()


from data_curate import DataCurate
from upload_dataset import UploadDataset

""" Data finetunning techniques: 
    1) Investigate
    2) Data clean/ Parse
    3) Visualise
    4) Data quality
    5) Curate
    6) Save
"""


current_dir = os.path.dirname(os.path.abspath(__file__))

class DataVisualize:

    def data_investigate(self, dataset):
        queryTypes = []
        queriesContentLength = []
        queries = []
        answersContentLength = []
        for datapoint in dataset:
            queryType = datapoint["QueryType"]
            if queryType !=  "Weather":
                queryTypes.append(queryType)
                queriesContentLength.append(len(datapoint["Query"]))
                queries.append(datapoint["Query"])
                answersContentLength.append(len(datapoint["Answer"]))

        return [queryTypes, queries, queriesContentLength, answersContentLength]


    def data_visualise(self, queryTypes, queries, queriesContentLength, answersContentLength):

        # Distribution of Query Types
        unique_types = list(set(queryTypes))
        counts = [queryTypes.count(qt) for qt in unique_types]

        fig_width = max(12, len(unique_types) * 0.8)
        plt.figure(figsize=(fig_width, 8))
        plt.subplot(1, 3, 1)
        x_pos = list(range(len(unique_types)))
        plt.bar(x_pos, counts, width=0.3, color='skyblue', edgecolor='black')
        plt.title("Distribution of Query Types")
        plt.xlabel("Query Type")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(x_pos, unique_types, rotation=45, ha='right', fontsize=8)
        plt.subplots_adjust(bottom=0.35, left=0.06, right=0.98)
        # plt.tight_layout()  
        
        # Distribution of Queries Content length
        plt.figure(figsize=(15, 6))
        plt.title(f"Lengths: Avg {sum(queriesContentLength)/len(queriesContentLength):,.0f} and highest {max(queriesContentLength):,}\n")
        plt.xlabel('Query Length (chars)')
        plt.ylabel('Count')
        plt.hist(queriesContentLength, rwidth=0.7, color="lightblue", bins=range(0, 1000, 10))
       
        # Distribution of Answers Content length
        plt.figure(figsize=(15, 6))
        plt.title(f"Lengths: Avg {sum(answersContentLength)/len(answersContentLength):,.0f} and highest {max(answersContentLength):,}\n")
        plt.xlabel('Answer Length (chars)')
        plt.ylabel('Count')
        plt.hist(answersContentLength, rwidth=0.7, color="lightblue", bins=range(0, 2000, 10))
       


        # -------------------------------
        # Duplicate words
        # -------------------------------
        words = []
        for q in queries:
            words.extend(q.lower().split()) 

        word_counts = Counter(words)
        # extract words that appear more than once
        duplicates = [word for word, count in word_counts.items() if count > 20]
        counts = [duplicates.count(qt) for qt in duplicates]
        print(duplicates)
        fig_width = max(12, len(duplicates) * 0.8)
        plt.figure(figsize=(fig_width, 8))
        plt.subplot(1, 3, 1)
        x_pos = list(range(len(duplicates)))
        plt.bar(x_pos, counts, width=0.3, color='skyblue', edgecolor='black')
        plt.title("Distribution of Query Types")
        plt.xlabel("Query Type")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(x_pos, duplicates, rotation=45, ha='right', fontsize=8)
        plt.subplots_adjust(bottom=0.35, left=0.06, right=0.98)
        
        plt.tight_layout()  
        plt.show()




def data_investigate(dataset):
    dataVisualize = DataVisualize()
    lengths, prices = dataVisualize.data_investigate(dataset)
    dataVisualize.data_visualise(lengths, prices)

def data_curate(visualise=True):
    dataCurate = DataCurate()
    # items = dDataCurate.getApiData()
    items = dataCurate.getData()
    if visualise:
        dataVisualize = DataVisualize()
        # sample = items[:5000]  # Sample first 5000 items for testing
        queryTypes, queries, queriesContentLength, answersContentLength = dataVisualize.data_investigate(items)
        dataVisualize.data_visualise(queryTypes, queries, queriesContentLength, answersContentLength)

    upload_dataset(items)


def upload_dataset(items):
        print(items[0])
        UploadDataset(items).upload()

if __name__ == "__main__":
    data_curate(False)
    


# Use this command to run : python src/data/data_visualize.py 