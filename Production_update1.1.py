import os
import re
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pickle
import logging
from datetime import datetime

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
        lr_params = {'normalize': [True, False],'fit_intercept': [True, False]} 
        lr_model = GridSearchCV(LinearRegression(), lr_params, cv=5, scoring='neg_mean_squared_error')
        lr_model.fit(X_train, y_train)
        return lr_model.best_estimator_
  
    

    def train_decision_tree(self, X_train, y_train):
        dt_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        dt_model = RandomizedSearchCV(DecisionTreeRegressor(), dt_params, cv=5, scoring='neg_mean_squared_error')
        dt_model.fit(X_train, y_train)
        return dt_model.best_estimator_

    def train_svm(self, X_train, y_train):
        svr_params = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'epsilon': [0.1, 0.5, 1]}
        svr = GridSearchCV(SVR(), svr_params, cv=5, scoring='neg_mean_squared_error')
        svr.fit(X_train, y_train)
        return svr.best_estimator_

    def train_random_forest(self, X_train, y_train):
        rfm_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        rfm = RandomizedSearchCV(RandomForestRegressor(), rfm_params, cv=5, scoring='neg_mean_squared_error')
        rfm.fit(X_train, y_train)
        return rfm.best_estimator_


class ModelEvaluator:
    def __init__(self):
        pass
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for training data
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        # Calculate metrics for testing data
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        # Logging
        self.log_metrics(model_name, mae_train, mse_train, r2_train, mae_test, mse_test, r2_test)
        
        return mae_train, mse_train, r2_train, mae_test, mse_test, r2_test
    
    def log_metrics(self, model_name, mae_train, mse_train, r2_train, mae_test, mse_test, r2_test):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_folder = 'logging'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log_file = os.path.join(log_folder, f'model_logfile_{current_time}.log')
        logging.basicConfig(filename=log_file, level=logging.INFO)
        logging.info("---------------------")
        logging.info(f"Model: {model_name}")
        logging.info("Metrics for training data:")
        logging.info("Mean Absolute Error: %f", mae_train)
        logging.info("Mean Squared Error: %f", mse_train)
        logging.info("R-squared: %f", r2_train)
        logging.info("\nMetrics for testing data:")
        logging.info("Mean Absolute Error: %f", mae_test)
        logging.info("Mean Squared Error: %f", mse_test)
        logging.info("R-squared: %f", r2_test)


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
    evaluator.evaluate_model(lr_model, X_train, y_train, X_test, y_test, 'Linear Regression')
    
    # Save linear regression model
    model_saver = ModelSaver()
    model_saver.save_model(lr_model, 'linear_regression_model')
    
    # Train decision tree model
    dt_model = trainer.train_decision_tree(X_train, y_train)
    
    # Evaluate decision tree model
    evaluator.evaluate_model(dt_model, X_train, y_train, X_test, y_test, 'Decision Tree')
    
    # Save decision tree model
    model_saver.save_model(dt_model, 'decision_tree_model')
    
    # Train SVM model
    svm_model = trainer.train_svm(X_train, y_train)
    
    # Evaluate SVM model
    evaluator.evaluate_model(svm_model, X_train, y_train, X_test, y_test, 'SVM')
    
    # Save SVM model
    model_saver.save_model(svm_model, 'svm_model')
    
    # Train Random Forest model
    rf_model = trainer.train_random_forest(X_train, y_train)
    
    # Evaluate Random Forest model
    evaluator.evaluate_model(rf_model, X_train, y_train, X_test, y_test, 'Random Forest')
    
    # Save Random Forest model
    model_saver.save_model(rf_model, 'random_forest_model')
