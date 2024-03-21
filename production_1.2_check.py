import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import pickle
import logging  # Add this import statement

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


class ModelTrainer:
    def __init__(self):
        pass
    
    def train_linear_regression(self, X_train, y_train):
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        return lr_model
  
    def train_decision_tree(self, X_train, y_train):
        dt_model = DecisionTreeRegressor()
        dt_model.fit(X_train, y_train)
        return dt_model


class ModelEvaluator:
    def __init__(self):
        pass
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Logging
        self.log_evaluation_results(model.__class__.__name__, mae, r2)
        
        return mae, r2
    
    def log_evaluation_results(self, model_name, mae, r2):
        # Create logging folder if it doesn't exist
        log_folder = 'logging'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        
        # Create log file path with model name and current time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_folder, f'{model_name}_{current_time}.log')
        
        # Configure logging to write to the specified file
        logging.basicConfig(filename=log_file, level=logging.INFO)
        
        # Log evaluation results
        logging.info("Model: %s", model_name)
        logging.info("Mean Absolute Error: %f", mae)
        logging.info("R-squared: %f", r2)

    


class ModelSaver:
    def __init__(self):
        pass
    
    def save_model(self, model, model_name):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = 'models'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path = os.path.join(folder_name, f'{model_name}_{current_time}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        return file_path

    def save_best_model(self, model, model_name):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = 'best_performing_model'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path = os.path.join(folder_name, f'{model_name}_{current_time}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        return file_path


if __name__ == "__main__":
    # Read data
    zooper_df = pd.read_csv('generated_dataset_v_0.1.8.csv')
    
    # Instantiate preprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_data(zooper_df)
    
    # Prepare data for training
    X = df.drop('speed', axis=1)  
    y = df['speed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
    
    # Instantiate model trainer
    trainer = ModelTrainer()
    
    # Train linear regression model
    lr_model = trainer.train_linear_regression(X_train, y_train)
    
    # Evaluate linear regression model
    evaluator = ModelEvaluator()
    lr_mae, lr_r2 = evaluator.evaluate_model(lr_model, X_test, y_test)
    
    # Save linear regression model
    model_saver = ModelSaver()
    lr_model_path = model_saver.save_model(lr_model, 'linear_regression_model')
    
    # Train decision tree model
    dt_model = trainer.train_decision_tree(X_train, y_train)
    
    # Evaluate decision tree model
    dt_mae, dt_r2 = evaluator.evaluate_model(dt_model, X_test, y_test)
    
    # Save decision tree model
    dt_model_path = model_saver.save_model(dt_model, 'decision_tree_model')

    # Determine the best performing model
    if lr_r2 > dt_r2:
        best_model = lr_model
        best_model_name = 'linear_regression_model'
    else:
        best_model = dt_model
        best_model_name = 'decision_tree_model'
    
    # Save the best performing model
    best_model_file_path = model_saver.save_best_model(best_model, best_model_name)
    
    print(f"The best performing model ({best_model_name}) has been saved to: {best_model_file_path}")
