

"""
@author: Naveen N G
@date: 1-10-2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datetime
from dataset_loader import DatasetLoader


START_YEAR = 2025


class DataCurate:

    def getApiData(self):

        current_year = datetime.date.today().year
        years_list = [year for year in range(START_YEAR, current_year)] 

        items = []
        for year in years_list:
            print(f'Data Loading for : year-{year} initiated')
            loader = DatasetLoader(year)
            data = loader.load(max_page=10, workers=6)
            print(f'Dataset Loading for : year-{year} completed')
            data.extend(data)
        return items
    
    def getData(self):
        # file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/kcc_data_2025.csv'))
        file_path = os.path.join(os.path.dirname(__file__), 'kcc_data.csv')
        loader = DatasetLoader()
        data_list = loader.load_data(file_path)
        return data_list
