
# Streamlit App: Enhanced MLB Strikeout Prop Dashboard with Full Pitcher Stats
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Safe & Verbose Pitcher Stats Scraper ---
def get_pitcher_stats():
    url = "https://www.baseball-reference.com/leagues/majors/2024-standard-pitching.shtml"
    try:
        tables = pd.read_html(url)
        st.write(f"✅ Found {len(tables)} tables.")
        for i, table in enumerate(tables):
            st.write(f"🔍 Table {i} Columns: {list(table.columns)}")
            if 'Player' in table.columns and 'SO9' in table.columns and 'IP' in table.columns:
                df = table[['Player', 'IP', 'SO9', 'BB9', 'H/9', 'HR/9', 'ERA', 'FIP', 'WHIP']].copy()
                df.columns = ['Pitcher', 'IP', 'K9', 'BB9', 'H9', 'HR9', 'ERA', 'FIP', 'WHIP']
                df['Pitcher'] = df['Pitcher'].str.replace(r'\.*', '', regex=True).str.strip()
                df = df.apply(pd.to_numeric, errors='ignore')
                return df
        st.warning("⚠️ No valid pitcher table found in Baseball-Reference scrape.")
        return pd.DataFrame(columns=['Pitcher', 'IP', 'K9', 'BB9', 'H9', 'HR9', 'ERA', 'FIP', 'WHIP'])
    except Exception as e:
        st.error(f"❌ Failed to scrape pitcher stats: {e}")
        return pd.DataFrame(columns=['Pitcher', 'IP', 'K9', 'BB9', 'H9', 'HR9', 'ERA', 'FIP', 'WHIP'])

# Dummy opponent team batting stats
def get_team_batting_stats():
    return {
        'Yankees': {'K%': 22.3, 'wRC+': 107},
        'Dodgers': {'K%': 20.9, 'wRC+': 115},
        'Astros': {'K%': 18.7, 'wRC+': 102},
        'Blue Jays': {'K%': 21.2, 'wRC+': 98},
    }

# --- Model Training ---
def train_model():
    data = pd.DataFrame({
        'K9': [9.5, 8.2, 10.1, 7.3, 11.0],
        'BB9': [2.5, 3.2, 2.1, 2.8, 1.7],
        'H9': [7.0, 8.5, 6.8, 9.1, 6.3],
        'HR9': [1.1, 1.3, 0.8, 1.5, 0.7],
        'ERA': [3.45, 4.12, 2.95, 4.65, 2.80],
        'FIP': [3.60, 4.01, 3.10, 4.80, 2.75],
        'WHIP': [1.12, 1.29, 1.03, 1.34, 0.98],
        'Opponent_K%': [25.0, 21.5, 27.2, 23.3, 29.1],
        'Opponent_wRC+': [96, 102, 89, 94, 78],
        'Predicted_Ks': [6.5, 5.0, 7.2, 4.3, 8.0]
    })
    X = data.drop(columns='Predicted_Ks')
    y = data['Projected_Ks']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- Streamlit App Layout ---
st.set_page_config(page_title="MLB Strikeout Prop Dashboard", layout="wide")
st.markdown("""
# 📈 Enhanced Pitcher Strikeout Predictor

This dashboard projects MLB pitcher strikeouts using real stats.
""")

options = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(4)]
selected_date = st.selectbox("Select Game Date", options)
st.markdown(f"**Selected date:** `{selected_date}`  
**Last updated:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

def get_pitchers_by_date(date):
    pitcher_stats = get_pitcher_stats()
    team_stats = get_team_batting_stats()
    if pitcher_stats.empty:
        return pd.DataFrame()

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher"
    try:
        games = requests.get(url).json().get('dates', [])[0].get('games', [])
    except:
        return pd.DataFrame()

    rows = []
    for game in games:
        home = game['teams']['home']['team']['name']
        away = game['teams']['away']['team']['name']
        for side in ['home', 'away']:
            pitcher = game['teams'][side].get('probablePitcher')
            if not pitcher:
                continue
            name = pitcher['fullName']
            team = home if side == 'home' else away
            opp = away if side == 'home' else home
            matchup = f"vs {opp}" if side == 'home' else f"@ {opp}"
            row = {'Pitcher': name, 'Team': team, 'Matchup': matchup}
            last_name = name.split()[-1].lower()
            if 'Pitcher' in pitcher_stats.columns:
                stat = pitcher_stats[pitcher_stats['Pitcher'].str.lower().str.contains(last_name)]
                if not stat.empty:
                    for col in ['K9', 'BB9', 'H9', 'HR9', 'ERA', 'FIP', 'WHIP', 'IP']:
                        row[col] = stat.iloc[0][col]
            opp_stats = team_stats.get(opp, {})
            row['Opponent_K%'] = opp_stats.get('K%', np.nan)
            row['Opponent_wRC+'] = opp_stats.get('wRC+', np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

model = train_model()
df = get_pitchers_by_date(selected_date)

features = ['K9', 'BB9', 'H9', 'HR9', 'ERA', 'FIP', 'WHIP', 'Opponent_K%', 'Opponent_wRC+']

if df.empty:
    st.warning("⚠️ No pitcher data found or available.")
else:
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=features, inplace=True)
    if df.empty:
        st.warning("⚠️ No usable pitcher rows after filtering.")
    else:
        df['Predicted_Ks'] = model.predict(df[features])
        display_cols = ['Pitcher', 'Team', 'Matchup'] + features + ['Predicted_Ks']
        df[features + ['Predicted_Ks']] = df[features + ['Predicted_Ks']].round(2)
        st.subheader("📊 Predicted Strikeouts")
        st.dataframe(df[display_cols].sort_values(by='Predicted_Ks', ascending=False).reset_index(drop=True), use_container_width=True)
