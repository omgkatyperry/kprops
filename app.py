
# Streamlit App: Pitcher Strikeout Prop Dashboard with Data Scraper
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Real Pitcher Stats Scraper from Baseball-Reference ---
def get_pitcher_stats():
    url = "https://www.baseball-reference.com/leagues/majors/2024-pitching-leaders.shtml"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if 'Name' in table.columns and 'SO9' in table.columns and 'IP' in table.columns:
                df = table[['Name', 'SO9', 'IP']].copy()
                df.columns = ['Pitcher', 'Avg_K_9', 'Innings_Pitched']
                df['Pitcher'] = df['Pitcher'].str.replace(r'\.*', '', regex=True).str.strip()
                df['Innings_Pitched'] = pd.to_numeric(df['Innings_Pitched'], errors='coerce')
                return df
        st.error("❌ No valid pitcher stat table found on Baseball-Reference.")
        return pd.DataFrame(columns=['Pitcher', 'Avg_K_9', 'Innings_Pitched'])
    except Exception as e:
        st.error(f"❌ Failed to scrape Baseball-Reference: {e}")
        return pd.DataFrame(columns=['Pitcher', 'Avg_K_9', 'Innings_Pitched'])

# --- Opponent Team Batting Stats (Static Placeholder until replaced) ---
def get_team_batting_stats():
    return {
        'Braves': {'K%': 21.3, 'BA': .258, 'OBP': .330, 'wRC+': 112},
        'Yankees': {'K%': 22.5, 'BA': .247, 'OBP': .325, 'wRC+': 108},
        'Blue Jays': {'K%': 20.2, 'BA': .249, 'OBP': .312, 'wRC+': 100},
        'Dodgers': {'K%': 22.1, 'BA': .260, 'OBP': .340, 'wRC+': 115}
    }

# --- Model Training ---
def train_model():
    data = pd.DataFrame({
        'Avg_K_9': [9.5, 8.2, 10.1, 7.3, 11.0],
        'Innings_Pitched': [6.0, 5.2, 6.1, 4.1, 7.0],
        'Opponent_K_Rate': [25.0, 21.5, 27.2, 23.3, 29.1],
        'Opponent_BA': [0.242, 0.260, 0.230, 0.250, 0.218],
        'Opponent_OBP': [0.315, 0.330, 0.298, 0.310, 0.275],
        'Opponent_WRC_Plus': [96, 102, 89, 94, 78],
        'Projected_Ks': [6.5, 5.0, 7.2, 4.3, 8.0]
    })
    X = data.drop(columns='Projected_Ks')
    y = data['Projected_Ks']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- App UI + Prediction Logic ---
st.set_page_config(page_title="MLB Strikeout Prop Dashboard", layout="wide")
st.title("⚾ Daily Pitcher Strikeout Model")

options = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(4)]
selected_date = st.selectbox("Select Game Date", options)
st.caption(f"Selected date: {selected_date} — Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_pitchers_by_date(date):
    pitcher_stats = get_pitcher_stats()
    team_stats = get_team_batting_stats()
    mlb_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher"
    try:
        games = requests.get(mlb_url).json().get('dates', [])[0].get('games', [])
    except:
        return pd.DataFrame()

    if pitcher_stats.empty:
        st.error("❌ No pitcher stats found from Baseball-Reference.")
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
            stat = pitcher_stats[pitcher_stats['Pitcher'].str.lower() == name.lower()]
            if not stat.empty:
                row['Avg_K_9'] = stat['Avg_K_9'].values[0]
                row['Innings_Pitched'] = stat['Innings_Pitched'].values[0]
            opp_stats = team_stats.get(opp, {})
            row['Opponent_K_Rate'] = opp_stats.get('K%', np.nan)
            row['Opponent_BA'] = opp_stats.get('BA', np.nan)
            row['Opponent_OBP'] = opp_stats.get('OBP', np.nan)
            row['Opponent_WRC_Plus'] = opp_stats.get('wRC+', np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

model = train_model()
pitchers_df = get_pitchers_by_date(selected_date)

features = [
    'Avg_K_9', 'Innings_Pitched', 'Opponent_K_Rate',
    'Opponent_BA', 'Opponent_OBP', 'Opponent_WRC_Plus'
]

if pitchers_df.empty:
    st.warning("No pitcher data available for the selected date.")
else:
    for col in features:
        if col in pitchers_df.columns:
            pitchers_df[col] = pd.to_numeric(pitchers_df[col], errors='coerce')
        else:
            pitchers_df[col] = np.nan

    pitchers_df.dropna(subset=features, inplace=True)

    if pitchers_df.empty:
        st.warning("No usable pitcher data found after filtering. Likely due to unmatched names or missing stats.")
    else:
        pitchers_df['Predicted_Ks'] = model.predict(pitchers_df[features])
        display_cols = ['Pitcher', 'Team', 'Matchup', 'Avg_K_9', 'Innings_Pitched',
                        'Opponent_K_Rate', 'Opponent_BA', 'Opponent_OBP', 'Opponent_WRC_Plus', 'Predicted_Ks']
        styled_df = pitchers_df[display_cols].copy()
        styled_df[features + ['Predicted_Ks']] = styled_df[features + ['Predicted_Ks']].round(2)
        st.dataframe(styled_df.sort_values(by='Predicted_Ks', ascending=False).reset_index(drop=True), use_container_width=True)
