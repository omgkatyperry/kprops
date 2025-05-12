
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor

def get_today_pitchers():
    return pd.DataFrame({
        'Pitcher': ['Zack Wheeler'],
        'Avg_K_9': [10.1],
        'Innings_Pitched': [6.0],
        'Opponent_K_Rate': [20.3],
        'Opponent_BA': [0.248],
        'Opponent_OBP': [0.312],
        'Opponent_WRC_Plus': [94],
        'Opponent_vs_Handedness_KRate': [19.8],
        'Umpire_K_Factor': [1.05],
        'Sportsbook_Line': [6.5]
    })

def train_model():
    train_data = pd.DataFrame({
        'Avg_K_9': [9.5, 8.2, 10.1, 7.3, 11.0],
        'Innings_Pitched': [6.0, 5.2, 6.1, 4.1, 7.0],
        'Opponent_K_Rate': [25.0, 21.5, 27.2, 23.3, 29.1],
        'Opponent_BA': [0.242, 0.260, 0.230, 0.250, 0.218],
        'Opponent_OBP': [0.315, 0.330, 0.298, 0.310, 0.275],
        'Opponent_WRC_Plus': [96, 102, 89, 94, 78],
        'Opponent_vs_Handedness_KRate': [26.1, 19.4, 28.5, 22.6, 30.3],
        'Umpire_K_Factor': [1.00, 0.97, 1.03, 0.95, 1.05],
        'Projected_Strikeouts': [6.5, 5.0, 7.2, 4.3, 8.0]
    })
    features = train_data.drop(columns='Projected_Strikeouts')
    target = train_data['Projected_Strikeouts']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    return model

def predict_props(data, model):
    features = ['Avg_K_9', 'Innings_Pitched', 'Opponent_K_Rate',
                'Opponent_BA', 'Opponent_OBP', 'Opponent_WRC_Plus',
                'Opponent_vs_Handedness_KRate', 'Umpire_K_Factor']
    data['Predicted_Ks'] = model.predict(data[features])
    data['Edge'] = data['Predicted_Ks'] - data['Sportsbook_Line']
    data['Confidence'] = data['Edge'].apply(
        lambda x: 'High' if abs(x) > 1.25 else 'Moderate' if abs(x) > 0.75 else 'Low'
    )
    return data

st.set_page_config(page_title="MLB Strikeout Prop Dashboard", layout="wide")
st.title("⚾ Daily Pitcher Strikeout Props")

data = get_today_pitchers()
model = train_model()
results = predict_props(data, model)

st.dataframe(results[['Pitcher', 'Predicted_Ks', 'Sportsbook_Line', 'Edge', 'Confidence']])

conf_filter = st.selectbox("Filter by Confidence", options=['All', 'High', 'Moderate', 'Low'])
if conf_filter != 'All':
    results = results[results['Confidence'] == conf_filter]
    st.dataframe(results[['Pitcher', 'Predicted_Ks', 'Sportsbook_Line', 'Edge', 'Confidence']])
