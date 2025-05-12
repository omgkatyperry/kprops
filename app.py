# Streamlit App: Pitcher Strikeout Prop Dashboard with Data Scraper
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Data Scraper from MLB's JSON API ---
def get_pitchers_by_date(selected_date):
    pitchers = []
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={selected_date}&hydrate=probablePitcher"
    response = requests.get(url)
    data = response.json()

    try:
        games = data['dates'][0]['games']
        for game in games:
            teams = game.get('teams', {})
            home_team = teams.get('home', {}).get('team', {}).get('name', 'Home')
            away_team = teams.get('away', {}).get('team', {}).get('name', 'Away')
            for side in ['home', 'away']:
                pitcher_info = teams.get(side, {}).get('probablePitcher')
                if pitcher_info:
                    pitcher_name = pitcher_info['fullName']
                    pitcher_team = home_team if side == 'home' else away_team
                    opponent_team = away_team if side == 'home' else home_team
                    pitcher_data = {
                        'Pitcher': pitcher_name,
                        'Team': pitcher_team,
                        'Matchup': f"{pitcher_team} vs {opponent_team}",
                        'Avg_K_9': np.random.uniform(7.5, 11.5),
                        'Innings_Pitched': np.random.uniform(5.0, 7.0),
                        'Opponent_K_Rate': np.random.uniform(18.0, 27.0),
                        'Opponent_BA': np.random.uniform(0.220, 0.270),
                        'Opponent_OBP': np.random.uniform(0.290, 0.340),
                        'Opponent_WRC_Plus': np.random.randint(85, 110),
                        'Opponent_vs_Handedness_KRate': np.random.uniform(18.0, 30.0),
                        'Umpire_K_Factor': np.random.uniform(0.95, 1.05),
                        'Sportsbook_Line': np.random.uniform(4.5, 7.5)
                    }
                    pitchers.append(pitcher_data)
    except Exception as e:
        print("Error parsing MLB API:", e)

    return pd.DataFrame(pitchers)

# --- Model Training (Mock Historical Data) ---
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

# --- Prediction Function ---
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

# --- Streamlit App ---
st.set_page_config(page_title="MLB Strikeout Prop Dashboard", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f9f9f9;}
        .css-18e3th9 {padding-top: 2rem;}
        .block-container {padding: 2rem;}
        .stDataFrame th, .stDataFrame td {text-align: center; font-size: 14px;}
    </style>
""", unsafe_allow_html=True)

st.title("⚾ Daily Pitcher Strikeout Props")

# Date picker dropdown
options = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(4)]
selected_date = st.selectbox("Select Game Date", options)
st.caption(f"Selected date: {selected_date} — Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load and predict
data = get_pitchers_by_date(selected_date)

if data.empty:
    st.warning("No starting pitchers were found for the selected date. MLB API may not have been updated yet.")
else:
    model = train_model()
    results = predict_props(data, model)

    # Optional filters
    conf_filter = st.selectbox("Filter by Confidence", options=['All', 'High', 'Moderate', 'Low'])
    if conf_filter != 'All':
        results = results[results['Confidence'] == conf_filter]

    # Display output
    display_cols = [
        'Pitcher', 'Team', 'Matchup', 'Predicted_Ks', 'Sportsbook_Line', 'Edge', 'Confidence',
        'Avg_K_9', 'Innings_Pitched', 'Opponent_K_Rate', 'Opponent_BA',
        'Opponent_OBP', 'Opponent_WRC_Plus', 'Opponent_vs_Handedness_KRate', 'Umpire_K_Factor'
    ]
    st.dataframe(results[display_cols].sort_values(by='Predicted_Ks', ascending=False).reset_index(drop=True))
