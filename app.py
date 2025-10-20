import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page config
st.set_page_config(page_title="NFLX Stock Predictor", layout="wide")

# Custom CSS for Netflix theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Model Definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# Cache functions
@st.cache_data
def load_data():
    """Load historical stock data."""
    df = pd.read_csv("NFLX_data.csv")
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
    df = df.drop(['Dividends', 'Stock Splits'], axis=1)
    return df

@st.cache_resource
def load_model(device):
    """Load the pre-trained model."""
    model = LSTM(input_size=4, hidden_size=128, num_layers=2, output_size=1, dropout=0.2)
    model.load_state_dict(torch.load('nflx.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# create sequences for LSTM
def create_sequences(data, window_size):
    """Create sequences for LSTM."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

def denormalize_predictions(scaled_preds, scaler, num_features):
    """Denormalize predictions."""
    dummy = np.zeros((len(scaled_preds), num_features))
    dummy[:, -1] = scaled_preds
    denormalized = scaler.inverse_transform(dummy)
    return denormalized[:, -1]

# App Info
st.markdown('<p class="main-header">Netflix Stock Price Predictor</p>', unsafe_allow_html=True)
#st.markdown('<p class="sub-header">Powered by LSTM Deep Learning</p>', unsafe_allow_html=True)

# Load data and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with st.spinner("Loading data and model..."):
    df = load_data()
    model = load_model(device)

# Prepare data (matching notebook exactly)
features = ['Open', 'High', 'Low', 'Volume']
features_data = df[features].values
target_data = df['Close'].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features_data)

window_size = 90
X, y = create_sequences(scaled_data, window_size)

train_size = int(len(X) * 0.8)
X_test, y_test = X[train_size:], y[train_size:]

# --- MAIN CONTENT ---

# Section 1: Model Performance Metrics
st.header("Model Performance on Test Data")
st.write("The model was trained on 80% of historical data and evaluated on the remaining 20%.")

col1, col2, col3, col4 = st.columns(4)

# Make predictions on test set
with torch.no_grad():
    test_inputs = torch.FloatTensor(X_test).to(device)
    test_preds_scaled = model(test_inputs).cpu().numpy().flatten()

predictions_denorm = denormalize_predictions(test_preds_scaled, scaler, 4)
actuals_denorm = denormalize_predictions(y_test, scaler, 4)

# Calculate metrics
mae = mean_absolute_error(actuals_denorm, predictions_denorm)
mse = mean_squared_error(actuals_denorm, predictions_denorm)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actuals_denorm - predictions_denorm) / actuals_denorm)) * 100

with col1:
    st.metric("MAE", f"${mae:,.2f}", delta=None, delta_color='normal')
with col2:
    st.metric("RMSE", f"${rmse:,.2f}", delta=None)
with col3:
    st.metric("MAPE", f"{mape:.2f}%", delta=None)
with col4:
    st.metric("Test Samples", f"{len(predictions_denorm)}", delta=None)


# Section 1.5 Trend information for dataset
st.header("📈 Trend Information for NFLX")

fig4 = plt.figure(figsize=(14,6))
sns.lineplot(x=df.index, y=df['Close'], label='Close Price', color='red', linewidth=1)
plt.grid()
plt.title('Close Price vs Dates for NFLX')
st.pyplot(fig4)


# Section 2: Predictions vs Actuals Visualization
st.header("📈 Predictions vs Actual Prices (Test Set)")

fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(actuals_denorm, label='Actual Prices', linewidth=2, color='#E50914')
ax1.plot(predictions_denorm, label='Predicted Prices', linewidth=2, alpha=0.7, color='#221f1f')
ax1.set_title('LSTM Predictions vs Actual Close Price (Test Set)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time Step', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

# Section 3: Scatter Plot
st.header("🎯 Prediction Accuracy Scatter Plot")

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=actuals_denorm, y=predictions_denorm, alpha=0.6, s=50)
ax2.plot([actuals_denorm.min(), actuals_denorm.max()], 
         [actuals_denorm.min(), actuals_denorm.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Prices ($)', fontsize=12)
ax2.set_ylabel('Predicted Prices ($)', fontsize=12)
ax2.set_title('Predicted vs Actual Prices', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

# Section 4: Future Predictions
st.header("Future Price Predictions")

col1, col2 = st.columns([1, 3])

with col1:
    days_to_predict = st.slider("Days to Predict", 1, 60, 30, 1)
    predict_button = st.button("🔮 Predict Future Prices", type="primary")

if predict_button:
    with st.spinner(f"Predicting next {days_to_predict} days..."):
        # Get last sequence
        last_sequence = scaled_data[-window_size:]
        future_predictions_scaled = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(days_to_predict):
                X_input = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
                pred = model(X_input).cpu().numpy()[0, 0]
                future_predictions_scaled.append(pred)
                
                # Update sequence
                new_row = current_sequence[-1].copy()
                new_row[-1] = pred
                current_sequence = np.vstack([current_sequence[1:], new_row])
        
        future_predictions = denormalize_predictions(np.array(future_predictions_scaled), scaler, 4)
        
        # Create future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='D')
        
        # Display results
        st.success(f"✅ Predictions generated for {days_to_predict} days!")
        
        # Show table
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close Price': future_predictions
        })
        future_df['Predicted Close Price'] = future_df['Predicted Close Price'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(future_df.set_index('Date'), use_container_width=True)
        
        
# Section 5: Model Information
with st.expander("Model Information"):
    st.markdown("""
    **Model Architecture:**
    - Type: LSTM (Long Short-Term Memory) Neural Network
    - Input Features: Open, High, Low, Volume
    - Hidden Size: 128 units
    - Layers: 2 LSTM layers
    - Dropout: 0.2
    - Total Parameters: 200,833
    
    **Training Details:**
    - Training Period: 10 years of historical data
    - Training Samples: 1,940 (80%)
    - Test Samples: 485 (20%)
    - Epochs: 100
    - Learning Rate: 0.01
    - Optimizer: Adam
    - Loss Function: MSE (Mean Squared Error)
    
    **Data Source:**
    - Yahoo Finance (yfinance)
    - Ticker: NFLX
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><small>⚠️ Disclaimer: This is for educational purposes only. Stock predictions are not financial advice.</small></p>
</div>
""", unsafe_allow_html=True)
