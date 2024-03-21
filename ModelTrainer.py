from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

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
