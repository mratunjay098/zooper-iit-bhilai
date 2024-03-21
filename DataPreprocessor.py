import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        pass
    
    def preprocess_data(self, dataframe):
        df = dataframe.copy()
        df.drop(columns=['check_point', 'traffic'], inplace=True, axis=1)
        df = df[~((df['traffic_signal'] == 1) & (df['speed'].between(0, 10)))]
        df.drop("traffic_signal", inplace=True, axis=1)
        
        def categorize_timestamp(value):
            if value in range(1, 9) or value in [23, 24]:
                return 1
            elif value in range(9, 12) or value in range(14, 17) or value in range(20, 23):
                return 3
            elif value in range(12, 14) or value in range(17, 20):
                return 2
            else:
                return None
        
        df['timestamp'] = df['timestamp'].apply(categorize_timestamp)
        
        replace_dict = {'NH': 1, 'SH': 2, 'MDR': 3, 'ODR': 4}
        df['road_hierarchy'].replace(replace_dict, inplace=True)
        
        return df
