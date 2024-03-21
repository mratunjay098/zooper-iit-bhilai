import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from DataPreprocessor import DataPreprocessor
from ModelTrainer import ModelTrainer
from ModelEvaluator import ModelEvaluator
from plot_utils import plot_predictions  # Assuming you have a module called plot_utils for plotting

def load_data_and_models():
    # Read data
    zooper_df = pd.read_csv('generated_dataset_v_0.1.8.csv')
    
    # Instantiate preprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_data(zooper_df)
    
    # Prepare data for training
    X = df.drop('speed', axis=1)  
    y = df['speed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
    
    # Load Linear Regression model
    with open('models/linear_regression_model_2024-02-17_18-49-51.pkl', 'rb') as file:
        lr_model = pickle.load(file)
    
    # Load Decision Tree model
    with open('models/decision_tree_model_2024-02-09_17-16-31.pkl', 'rb') as file:
        dt_model = pickle.load(file)
    
    return X_test, y_test, lr_model, dt_model

if __name__ == "__main__":
    # Load data and models
    X_test, y_test, lr_model, dt_model = load_data_and_models()

    # Make predictions
    lr_predictions = lr_model.predict(X_test)
    dt_predictions = dt_model.predict(X_test)

    # Plotting
    plot_predictions(y_test, lr_predictions, model_name='Linear Regression')
    plot_predictions(y_test, dt_predictions, model_name='Decision Tree')
