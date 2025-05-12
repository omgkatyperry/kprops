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

# --- Final Placeholder Output (to be replaced with actual model integration) ---
st.info("Live model and prediction output coming shortly. Placeholder displayed below:")
df = pd.DataFrame({
    "Pitcher": ["Gerrit Cole", "Kevin Gausman"],
    "DK Line": [7.5, 6.5],
    "FD Line": [7.0, 6.0],
    "B365 Line": [7.5, 6.5],
    "Opponent": ["@ Red Sox", "vs Yankees"]
})
st.dataframe(df)
