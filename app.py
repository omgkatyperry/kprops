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

# --- Streamlit App UI + Prediction Logic ---
st.set_page_config(page_title="MLB Strikeout Prop Dashboard", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #eee;
        }
        .stApp {
            padding: 2rem;
        }
        .stDataFrame tbody tr:hover {
            background-color: #222 !important;
        }
        .stDataFrame thead tr th {
            background-color: #333;
            color: #fff;
        }
        .stSelectbox label, .stCaption, .stSubheader, .stTitle {
            color: #ddd;
        }
    </style>
""", unsafe_allow_html=True)
st.title("⚾ Daily Pitcher Strikeout Model")

options = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(4)]
selected_date = st.selectbox("Select Game Date", options)
st.caption(f"Selected date: {selected_date} — Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_pitchers_by_date(date):
    pitcher_stats = get_fangraphs_pitcher_stats()
    team_stats = get_team_batting_stats()
    mlb_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher"
    try:
        games = requests.get(mlb_url).json().get('dates', [])[0].get('games', [])
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
            stat = pitcher_stats[pitcher_stats['Pitcher'].str.contains(name.split()[-1], case=False)]
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

# DEBUG: Show raw pitcher data before filtering
st.subheader("🔍 Raw Pitcher Data Before Filtering")
st.dataframe(pitchers_df)


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
        try:
            pitchers_df['Predicted_Ks'] = model.predict(pitchers_df[features])

            display_cols = [
                'Pitcher', 'Team', 'Matchup',
                'Avg_K_9', 'Innings_Pitched', 'Opponent_K_Rate',
                'Opponent_BA', 'Opponent_OBP', 'Opponent_WRC_Plus',
                'Predicted_Ks'
            ]

            styled_df = pitchers_df[display_cols].copy()
            styled_df[display_cols[3:]] = styled_df[display_cols[3:]].round(2)

            st.dataframe(
                styled_df.sort_values(by='Predicted_Ks', ascending=False).reset_index(drop=True),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
