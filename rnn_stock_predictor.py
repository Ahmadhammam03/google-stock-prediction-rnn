"""
Google Stock Price Prediction using RNN (LSTM)
Author: Ahmad Hammam
Date: 2024

A Recurrent Neural Network implementation for predicting Google stock prices
using LSTM layers with time series data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import warnings
warnings.filterwarnings('ignore')

class StockPricePredictor:
    """
    A class for predicting stock prices using LSTM neural networks.
    """
    
    def __init__(self, time_steps=60, lstm_units=50, dropout_rate=0.2):
        """
        Initialize the stock price predictor.
        
        Args:
            time_steps (int): Number of previous days to use for prediction
            lstm_units (int): Number of LSTM units per layer
            dropout_rate (float): Dropout rate for regularization
        """
        self.time_steps = time_steps
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.training_set_scaled = None
        
    def load_and_preprocess_data(self, train_file, test_file):
        """
        Load and preprocess the training and testing data.
        
        Args:
            train_file (str): Path to training CSV file
            test_file (str): Path to testing CSV file
        """
        print("Loading and preprocessing data...")
        
        # Load training data
        dataset_train = pd.read_csv(train_file)
        self.training_set = dataset_train.iloc[:, 1:2].values
        
        # Load test data
        dataset_test = pd.read_csv(test_file)
        self.real_stock_price = dataset_test.iloc[:, 1:2].values
        
        # Scale the training data
        self.training_set_scaled = self.scaler.fit_transform(self.training_set)
        
        print(f"Training data shape: {self.training_set.shape}")
        print(f"Test data shape: {self.real_stock_price.shape}")
        
    def create_sequences(self):
        """
        Create training sequences for the LSTM model.
        """
        print("Creating training sequences...")
        
        X_train = []
        y_train = []
        
        for i in range(self.time_steps, len(self.training_set_scaled)):
            X_train.append(self.training_set_scaled[i-self.time_steps:i, 0])
            y_train.append(self.training_set_scaled[i, 0])
            
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)
        
        # Reshape for LSTM input
        self.X_train = np.reshape(self.X_train, 
                                 (self.X_train.shape[0], self.X_train.shape[1], 1))
        
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        
    def build_model(self):
        """
        Build the LSTM model architecture.
        """
        print("Building LSTM model...")
        
        self.model = Sequential()
        
        # First LSTM layer with Dropout regularization
        self.model.add(LSTM(units=self.lstm_units, 
                           return_sequences=True, 
                           input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer with Dropout
        self.model.add(LSTM(units=self.lstm_units, return_sequences=True))
        self.model.add(Dropout(self.dropout_rate))
        
        # Third LSTM layer with Dropout
        self.model.add(LSTM(units=self.lstm_units, return_sequences=True))
        self.model.add(Dropout(self.dropout_rate))
        
        # Fourth LSTM layer with Dropout
        self.model.add(LSTM(units=self.lstm_units))
        self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("Model architecture:")
        self.model.summary()
        
    def train_model(self, epochs=100, batch_size=32):
        """
        Train the LSTM model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        print(f"Training model for {epochs} epochs...")
        
        history = self.model.fit(self.X_train, self.y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=1)
        
        print("Training completed!")
        return history
        
    def prepare_test_data(self, train_file, test_file):
        """
        Prepare test data for prediction.
        
        Args:
            train_file (str): Path to training CSV file
            test_file (str): Path to testing CSV file
        """
        print("Preparing test data...")
        
        # Load datasets
        dataset_train = pd.read_csv(train_file)
        dataset_test = pd.read_csv(test_file)
        
        # Combine datasets
        dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
        
        # Get inputs for prediction
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - self.time_steps:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)
        
        # Create test sequences
        X_test = []
        for i in range(self.time_steps, self.time_steps + len(dataset_test)):
            X_test.append(inputs[i-self.time_steps:i, 0])
            
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        return X_test
        
    def make_predictions(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test (np.array): Test data sequences
            
        Returns:
            np.array: Predicted stock prices
        """
        print("Making predictions...")
        
        predicted_stock_price = self.model.predict(X_test)
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)
        
        return predicted_stock_price
        
    def visualize_results(self, predicted_prices, save_path=None):
        """
        Visualize the prediction results.
        
        Args:
            predicted_prices (np.array): Predicted stock prices
            save_path (str): Path to save the visualization
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.real_stock_price, color='red', label='Real Google Stock Price')
        plt.plot(predicted_prices, color='blue', label='Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction using RNN (LSTM)', fontsize=16)
        plt.xlabel('Time (Trading Days)', fontsize=12)
        plt.ylabel('Google Stock Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            
        plt.show()
        
    def calculate_metrics(self, predicted_prices):
        """
        Calculate prediction accuracy metrics.
        
        Args:
            predicted_prices (np.array): Predicted stock prices
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        mse = mean_squared_error(self.real_stock_price, predicted_prices)
        mae = mean_absolute_error(self.real_stock_price, predicted_prices)
        rmse = np.sqrt(mse)
        
        print("\n" + "="*50)
        print("PREDICTION METRICS")
        print("="*50)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print("="*50)
        
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_saved_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")

def main():
    """
    Main function to run the stock price prediction pipeline.
    """
    print("="*60)
    print("GOOGLE STOCK PRICE PREDICTION USING RNN (LSTM)")
    print("="*60)
    
    # Initialize predictor
    predictor = StockPricePredictor(time_steps=60, lstm_units=50, dropout_rate=0.2)
    
    # File paths
    train_file = 'data/Google_Stock_Price_Train.csv'
    test_file = 'data/Google_Stock_Price_Test.csv'
    
    # Check if files exist
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Error: Dataset files not found!")
        print("Please ensure the following files are in the current directory:")
        print("- Google_Stock_Price_Train.csv")
        print("- Google_Stock_Price_Test.csv")
        return
    
    try:
        # Step 1: Load and preprocess data
        predictor.load_and_preprocess_data(train_file, test_file)
        
        # Step 2: Create training sequences
        predictor.create_sequences()
        
        # Step 3: Build model
        predictor.build_model()
        
        # Step 4: Train model
        history = predictor.train_model(epochs=100, batch_size=32)
        
        # Step 5: Save trained model
        predictor.save_model('models/google_stock_rnn_model.h5')
        
        # Step 6: Prepare test data and make predictions
        X_test = predictor.prepare_test_data(train_file, test_file)
        predicted_prices = predictor.make_predictions(X_test)
        
        # Step 7: Visualize results
        predictor.visualize_results(predicted_prices, 'visualizations/prediction_results.png')
        
        # Step 8: Calculate metrics
        predictor.calculate_metrics(predicted_prices)
        
        print("\n🎉 Stock price prediction completed successfully!")
        print("📊 Check the visualization and model files in their respective folders.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()