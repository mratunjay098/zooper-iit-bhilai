import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
