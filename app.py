# Streamlit App: Pitcher Strikeout Prop Dashboard with Data Scraper
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Real Pitcher K/9 and IP Scraper from FanGraphs ---
def get_fangraphs_pitcher_stats():
    url = "https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=0&type=1&season=2024&month=0&season1=2024&ind=0"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if 'Name' in table.columns and 'K/9' in table.columns and 'IP' in table.columns:
                df = table[['Name', 'K/9', 'IP']].copy()
                df.columns = ['Pitcher', 'Avg_K_9', 'Innings_Pitched']
                df['Pitcher'] = df['Pitcher'].str.strip()
                df['Innings_Pitched'] = pd.to_numeric(df['Innings_Pitched'], errors='coerce') / 30
                return df
    except Exception as e:
        print("Failed to scrape FanGraphs:", e)
        return pd.DataFrame(columns=['Pitcher', 'Avg_K_9', 'Innings_Pitched'])

# --- Opponent Team Batting Stats (Live from FanGraphs) ---
def get_team_batting_stats():
    url = "https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&type=8&season=2024&month=0&season1=2024&ind=0"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if 'Team' in table.columns and 'K%' in table.columns and 'AVG' in table.columns:
                df = table[['Team', 'K%', 'AVG', 'OBP', 'wRC+']].copy()
                df.columns = ['Team', 'K%', 'BA', 'OBP', 'wRC+']
                df['Team'] = df['Team'].str.strip()
                team_stats = {}
                for _, row in df.iterrows():
                    team_stats[row['Team']] = {
                        'K%': float(row['K%']),
                        'BA': float(row['BA']),
                        'OBP': float(row['OBP']),
                        'wRC+': int(row['wRC+'])
                    }
                return team_stats
    except Exception as e:
        print("Failed to scrape team batting stats from FanGraphs:", e)
        return {}

# --- Simulated Umpire K Factor Map ---
def get_umpire_k_factors():
    return {
        'James Hoye': 1.05,
        'Laz Diaz': 0.96,
        'Pat Hoberg': 1.01,
        'Mark Wegner': 1.03,
        'Tripp Gibson': 0.98,
    }

# --- Scheduled Umpire Mapping ---
def get_scheduled_umpire(home_team, away_team):
    try:
        schedule_url = "https://www.rotowire.com/baseball/mlb-lineups.php"
        response = requests.get(schedule_url, headers={"User-Agent": "Mozilla/5.0"})
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        matchups = soup.find_all('div', class_='lineup__card')
        for matchup in matchups:
            teams = matchup.find('div', class_='lineup__teams').text.strip()
            ump_info = matchup.find('div', class_='lineup__note')
            if ump_info and 'Umpire:' in ump_info.text:
                text = ump_info.text.strip()
                ump_line = [line for line in text.split('\n') if 'Umpire:' in line]
                if ump_line:
                    ump_name = ump_line[0].split('Umpire:')[-1].strip()
                    if home_team in teams and away_team in teams:
                        return ump_name
        return None
    except Exception as e:
        print("Failed to get scheduled umpire:", e)
        return None

# --- Odds Aggregator ---
def get_strikeout_odds():
    url = "https://www.oddsboom.com/mlb/strikeouts"
    odds_data = {}
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        for table in tables:
            if 'Player' in table.columns:
                for _, row in table.iterrows():
                    name = str(row['Player']).strip()
                    odds_data[name] = {
                        'DK': float(row.get('DraftKings', np.nan)),
                        'FD': float(row.get('FanDuel', np.nan)),
                        'B365': float(row.get('Bet365', np.nan))
                    }
        return odds_data
    except Exception as e:
        print("Failed to scrape strikeout props:", e)
        return {}

# --- Prediction Model ---
def train_model():
    data = pd.DataFrame({
        'Avg_K_9': [9.5, 8.2, 10.1, 7.3, 11.0],
        'Innings_Pitched': [6.0, 5.2, 6.1, 4.1, 7.0],
        'Opponent_K_Rate': [25.0, 21.5, 27.2, 23.3, 29.1],
        'Opponent_BA': [0.242, 0.260, 0.230, 0.250, 0.218],
        'Opponent_OBP': [0.315, 0.330, 0.298, 0.310, 0.275],
        'Opponent_WRC_Plus': [96, 102, 89, 94, 78],
        'Umpire_K_Factor': [1.00, 0.97, 1.03, 0.95, 1.05],
        'Projected_Ks': [6.5, 5.0, 7.2, 4.3, 8.0]
    })
    X = data.drop(columns='Projected_Ks')
    y = data['Projected_Ks']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- Streamlit UI ---
st.set_page_config(page_title="MLB Strikeout Prop Dashboard", layout="wide")
st.title("⚾ Daily Pitcher Strikeout Props")

options = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(4)]
selected_date = st.selectbox("Select Game Date", options)

st.caption(f"Selected date: {selected_date} — Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Generate and Display Model Predictions ---
from sklearn.exceptions import NotFittedError

# Placeholder function to gather real pitcher/matchup data
def get_pitchers_by_date(date):
    pitcher_stats = get_fangraphs_pitcher_stats()
    team_stats = get_team_batting_stats()
    umpire_map = get_umpire_k_factors()
    odds_data = get_strikeout_odds()

    mlb_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher"
    try:
        games = requests.get(mlb_url).json().get('dates', [])[0].get('games', [])
    except:
        return pd.DataFrame()

    rows = []
    for game in games:
        home = game['teams']['home']['team']['name']
        away = game['teams']['away']['team']['name']
        ump_name = get_scheduled_umpire(home, away)
        kfactor = umpire_map.get(ump_name, 1.00)

        for side in ['home', 'away']:
            pitcher = game['teams'][side].get('probablePitcher')
            if not pitcher: continue
            name = pitcher['fullName']
            team = home if side == 'home' else away
            opp = away if side == 'home' else home
            matchup = f"vs {opp}" if side == 'home' else f"@ {opp}"

            row = {'Pitcher': name, 'Team': team, 'Matchup': matchup, 'Umpire_K_Factor': kfactor}

            stat = pitcher_stats[pitcher_stats['Pitcher'].str.contains(name.split()[-1], case=False)]
            if not stat.empty:
                row['Avg_K_9'] = stat['Avg_K_9'].values[0]
                row['Innings_Pitched'] = stat['Innings_Pitched'].values[0]

            opp_stats = team_stats.get(opp, {})
            row['Opponent_K_Rate'] = opp_stats.get('K%', np.nan)
            row['Opponent_BA'] = opp_stats.get('BA', np.nan)
            row['Opponent_OBP'] = opp_stats.get('OBP', np.nan)
            row['Opponent_WRC_Plus'] = opp_stats.get('wRC+', np.nan)

            odds = odds_data.get(name, {})
            row['DK'] = odds.get('DK', np.nan)
            row['FD'] = odds.get('FD', np.nan)
            row['B365'] = odds.get('B365', np.nan)

            rows.append(row)
    return pd.DataFrame(rows)

# Train model and get pitcher data
model = train_model()
pitchers_df = get_pitchers_by_date(selected_date)

# Ensure required features are present and numeric
features = ['Avg_K_9', 'Innings_Pitched', 'Opponent_K_Rate', 'Opponent_BA', 'Opponent_OBP', 'Opponent_WRC_Plus', 'Umpire_K_Factor']
for col in features:
    pitchers_df[col] = pd.to_numeric(pitchers_df[col], errors='coerce')
pitchers_df.dropna(subset=features, inplace=True)

try:
    # Predict strikeouts
    pitchers_df['Predicted_Ks'] = model.predict(pitchers_df[features])
    pitchers_df['Edge_vs_DK'] = pitchers_df['Predicted_Ks'] - pitchers_df['DK']
    pitchers_df['Edge_vs_FD'] = pitchers_df['Predicted_Ks'] - pitchers_df['FD']
    pitchers_df['Edge_vs_B365'] = pitchers_df['Predicted_Ks'] - pitchers_df['B365']

    # Display
    display_cols = ['Pitcher', 'Team', 'Matchup', 'Predicted_Ks', 'DK', 'FD', 'B365', 'Edge_vs_DK', 'Edge_vs_FD', 'Edge_vs_B365']
    st.dataframe(pitchers_df[display_cols].sort_values(by='Predicted_Ks', ascending=False).reset_index(drop=True))

except NotFittedError:
    st.error("Model could not be fitted. Check training data.")
