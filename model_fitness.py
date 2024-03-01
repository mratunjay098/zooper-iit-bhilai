import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from DataPreprocessor import DataPreprocessor
from ModelTrainer import ModelTrainer
from ModelEvaluator import ModelEvaluator

def load_data():
    # Read data
    zooper_df = pd.read_csv('generated_dataset_v_0.1.8.csv')
    
    # Instantiate preprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_data(zooper_df)
    
    # Prepare data for training
    X = df.drop('speed', axis=1)  
    y = df['speed']
    
    return X, y

def plot_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, valid_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
                                                              scoring='neg_mean_squared_error', cv=5)

    train_scores_mean = -np.mean(train_scores, axis=1)
    valid_scores_mean = -np.mean(valid_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation error")
    plt.title(f"{model_name} Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="best")
    plt.grid(True)

    # Indicate whether the model is overfitting
    if (train_scores_mean[-1] < valid_scores_mean[-1]) or (np.mean(train_scores_mean) < np.mean(valid_scores_mean)):
        plt.text(train_sizes[-1], valid_scores_mean[-1], "Overfitting", fontsize=12, color='red', ha='center', va='bottom')

    plt.show()

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
    
    # Load Linear Regression model
    with open('models/linear_regression_model_2024-02-17_18-49-51.pkl', 'rb') as file:
        lr_model = pickle.load(file)
    
    # Load Decision Tree model
    with open('models/decision_tree_model_2024-02-09_17-16-31.pkl', 'rb') as file:
        dt_model = pickle.load(file)
    
    # Plot learning curves
    plot_learning_curve(lr_model, X_train, y_train, model_name='Linear Regression')
    plot_learning_curve(dt_model, X_train, y_train, model_name='Decision Tree')
