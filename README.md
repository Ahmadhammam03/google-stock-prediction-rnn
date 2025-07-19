# Google Stock Price Prediction using RNN (LSTM) 📈🤖

A Recurrent Neural Network implementation using LSTM layers to predict Google stock prices based on historical data. This project demonstrates time series forecasting capabilities with deep learning.

## 🎯 Project Overview

This project uses a multi-layer LSTM (Long Short-Term Memory) network to predict future Google stock prices based on 60 previous trading days. The model learns patterns from historical stock data to make informed predictions about future price movements.

## 🏗️ Model Architecture

```
RNN Architecture:
├── LSTM Layer 1 (50 units, return_sequences=True)
├── Dropout (0.2)
├── LSTM Layer 2 (50 units, return_sequences=True)  
├── Dropout (0.2)
├── LSTM Layer 3 (50 units, return_sequences=True)
├── Dropout (0.2)
├── LSTM Layer 4 (50 units)
├── Dropout (0.2)
└── Dense Output Layer (1 unit)
```

## 📊 Dataset Information

- **Training Data**: 1,258 trading days (Google stock prices 2012-2016)
- **Test Data**: 20 trading days (January 2017)
- **Features**: Open, High, Low, Close, Volume
- **Target**: Stock opening price prediction
- **Time Window**: 60 days for prediction

## 🚀 Key Features

- **Multi-layer LSTM**: 4 LSTM layers with 50 units each for complex pattern recognition
- **Dropout Regularization**: 20% dropout rate to prevent overfitting
- **Time Series Processing**: 60-day sliding window approach
- **Feature Scaling**: MinMax normalization for optimal training
- **Real-time Prediction**: Can predict next day's opening price
- **Visualization**: Beautiful prediction vs actual price comparison

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
TensorFlow/Keras
NumPy
Pandas
Matplotlib
Scikit-learn
```

### Install Dependencies
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### Clone Repository
```bash
git clone https://github.com/Ahmadhammam03/google-stock-prediction-rnn.git
cd google-stock-prediction-rnn
```

## 📁 Project Structure

```
google-stock-prediction-rnn/
│
├── rnn.ipynb                           # Main Jupyter notebook
├── Google_Stock_Price_Train.csv        # Training dataset (2012-2016)
├── Google_Stock_Price_Test.csv         # Testing dataset (Jan 2017)
├── rnn_stock_predictor.py              # Python script version
├── models/
│   └── google_stock_rnn_model.h5       # Trained model
├── visualizations/
│   └── prediction_results.png          # Results visualization
├── README.md
└── requirements.txt
```

## 🎯 Usage

### Training the Model
```python
# Run the Jupyter notebook
jupyter notebook rnn.ipynb

# Or run the Python script
python rnn_stock_predictor.py
```

### Making Predictions
```python
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = load_model('models/google_stock_rnn_model.h5')

# Prepare your data (60 previous days)
# prediction = model.predict(scaled_data)
# predicted_price = scaler.inverse_transform(prediction)
```

## 📈 Model Performance

- **Training Loss**: Converged to ~0.0015 after 100 epochs
- **Architecture**: 4-layer LSTM with dropout regularization
- **Prediction Window**: 60-day lookback period
- **Validation**: Tested on January 2017 data

### Results Visualization
The model shows strong correlation between predicted and actual stock prices, capturing major trends and price movements effectively.

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|--------|
| LSTM Units | 50 per layer |
| Number of Layers | 4 |
| Dropout Rate | 0.2 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Epochs | 100 |
| Batch Size | 32 |
| Time Steps | 60 |

## 📊 Data Preprocessing

1. **Feature Selection**: Used 'Open' price as primary feature
2. **Normalization**: MinMax scaling to range [0,1]
3. **Sequence Creation**: 60-day sliding windows
4. **Train-Test Split**: 2012-2016 for training, Jan 2017 for testing
5. **Reshaping**: 3D tensor format for LSTM input

## 🔮 Future Improvements

- [ ] Add multiple features (Volume, High, Low, Close)
- [ ] Implement attention mechanisms
- [ ] Add sentiment analysis from news data
- [ ] Create real-time prediction pipeline
- [ ] Implement ensemble methods
- [ ] Add technical indicators as features
- [ ] Deploy as web application
- [ ] Add confidence intervals for predictions

## 📚 Technical Concepts

### LSTM Networks
Long Short-Term Memory networks are designed to handle long-term dependencies in sequential data, making them perfect for stock price prediction.

### Time Series Forecasting
The model uses historical patterns to predict future values, considering:
- Temporal dependencies
- Market trends
- Price volatility patterns

### Dropout Regularization
Prevents overfitting by randomly setting 20% of neurons to zero during training.

## ⚠️ Disclaimer

This project is for educational purposes only. Stock price prediction is extremely challenging and this model should NOT be used for actual trading decisions. Past performance does not guarantee future results.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ahmad Hammam**
- GitHub: [@Ahmadhammam03](https://github.com/Ahmadhammam03)
- LinkedIn: [Ahmad Hammam](https://www.linkedin.com/in/ahmad-hammam-1561212b2)

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Google Finance for providing historical stock data
- Deep learning community for RNN/LSTM research
- Time series forecasting literature

## 📊 Model Insights

### Why LSTM for Stock Prediction?
- **Memory Cells**: Can remember important information over long periods
- **Gradient Flow**: Solves vanishing gradient problem in traditional RNNs
- **Pattern Recognition**: Excellent at finding complex temporal patterns
- **Non-linear Modeling**: Captures complex market dynamics

### Training Process
1. Data preprocessing and normalization
2. Sequence generation with 60-day windows
3. Multi-layer LSTM training with dropout
4. Model validation on unseen test data
5. Performance evaluation and visualization

---

⭐ **If you found this project helpful, please give it a star!** ⭐