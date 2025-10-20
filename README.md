# NFLX_Stock_Prediction
Contains a notebook and stramlit app used to predict NFLX stock price based on rolling data for 10 years and a LSTM network.

## Objective
The objective of this project was to introduce myself to the architecture of an LSTM model, and see how it is implemented in Pytorch. To do this, I pulled Netflix stock data for the last ten years from Yahoo Finance using yfinance, performed some exploratory data analysis, and trained an LSTM model to predict future close stock priced based on a sliding window of 90 days. I presented my findings on a Streamlit application which showcases trend information for the data and lets users predict future days up to 60 days.

## Technologies Used
- Python
- Jupyter
- Streamlit
- Pytorch
