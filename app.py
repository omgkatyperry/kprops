
# Streamlit App: Strikeout Predictor using CSV-Based Pitcher Data
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Load Pre-saved CSV Pitcher Stats ---
@st.cache_data
def get_pitcher_stats():
    try:
        url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/mlb-elo/mlb_elo.csv"  # TEMP placeholder for structure
        df = pd.read_csv(url)
        df = df.rename(columns=str.lower)
        df['pitcher'] = df['team1']
        df['ip'] = np.random.uniform(4.0, 7.0, size=len(df))
        df['k9'] = np.random.uniform(6.0, 12.0, size=len(df))
        df['bb9'] = np.random.uniform(1.5, 4.0, size=len(df))
        df['h9'] = np.random.uniform(6.0, 9.0, size=len(df))
        df['hr9'] = np.random.uniform(0.5, 2.0, size=len(df))
        df['era'] = np.random.uniform(2.5, 5.0, size=len(df))
        df['fip'] = np.random.uniform(2.8, 5.2, size=len(df))
        df['whip'] = np.random.uniform(1.0, 1.4, size=len(df))
        return df[['pitcher', 'ip', 'k9', 'bb9', 'h9', 'hr9', 'era', 'fip', 'whip']]
    except Exception as e:
        st.error(f"❌ Failed to load pitcher stats: {e}")
        return pd.DataFrame(columns=['pitcher', 'ip', 'k9', 'bb9', 'h9', 'hr9', 'era', 'fip', 'whip'])

def get_team_batting_stats():
    return {
        'Yankees': {'K%': 22.3, 'wRC+': 107},
        'Dodgers': {'K%': 20.9, 'wRC+': 115},
        'Astros': {'K%': 18.7, 'wRC+': 102},
        'Blue Jays': {'K%': 21.2, 'wRC+': 98},
    }

def train_model():
    data = pd.DataFrame({
        'k9': [9.5, 8.2, 10.1, 7.3, 11.0],
        'bb9': [2.5, 3.2, 2.1, 2.8, 1.7],
        'h9': [7.0, 8.5, 6.8, 9.1, 6.3],
        'hr9': [1.1, 1.3, 0.8, 1.5, 0.7],
        'era': [3.45, 4.12, 2.95, 4.65, 2.80],
        'fip': [3.60, 4.01, 3.10, 4.80, 2.75],
        'whip': [1.12, 1.29, 1.03, 1.34, 0.98],
        'Opponent_K%': [25.0, 21.5, 27.2, 23.3, 29.1],
        'Opponent_wRC+': [96, 102, 89, 94, 78],
        'Predicted_Ks': [6.5, 5.0, 7.2, 4.3, 8.0]
    })
    X = data.drop(columns='Predicted_Ks')
    y = data['Predicted_Ks']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

st.set_page_config(page_title="CSV-Based Strikeout Predictor", layout="wide")
st.title("📈 Strikeout Predictor (CSV Pitcher Stats)")

options = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(4)]
selected_date = st.selectbox("Select Game Date", options)

@st.cache_data
def get_dummy_schedule(date):
    pitchers = get_pitcher_stats()
    subset = pitchers.sample(n=min(10, len(pitchers)), random_state=42).copy()
    subset['Team'] = np.random.choice(['Yankees', 'Dodgers', 'Blue Jays'], size=len(subset))
    subset['Opponent'] = np.random.choice(['Astros', 'Rays', 'Twins'], size=len(subset))
    subset['Matchup'] = ["vs " + opp for opp in subset['Opponent']]
    return subset

model = train_model()
df = get_dummy_schedule(selected_date)
team_stats = get_team_batting_stats()

if df.empty:
    st.warning("⚠️ No pitcher schedule data found.")
else:
    for opp in df['Opponent'].unique():
        if opp not in team_stats:
            team_stats[opp] = {'K%': 22.0, 'wRC+': 100}

    df['Opponent_K%'] = df['Opponent'].map(lambda t: team_stats[t]['K%'])
    df['Opponent_wRC+'] = df['Opponent'].map(lambda t: team_stats[t]['wRC+'])

    features = ['k9', 'bb9', 'h9', 'hr9', 'era', 'fip', 'whip', 'Opponent_K%', 'Opponent_wRC+']
    df = df.dropna(subset=features)
    df['Predicted_Ks'] = model.predict(df[features])
    df[features + ['Predicted_Ks']] = df[features + ['Predicted_Ks']].round(2)

    st.subheader("📊 Predicted Strikeouts")
    st.dataframe(df[['pitcher', 'Team', 'Matchup'] + features + ['Predicted_Ks']].sort_values(by='Predicted_Ks', ascending=False), use_container_width=True)
