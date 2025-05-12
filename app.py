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
                df['Innings_Pitched'] = pd.to_numeric(df['Innings_Pitched'], errors='coerce') / 30  # Est. IP per start
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

# --- Scheduled Umpire Mapping (Manual Example) ---
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
                ump_name = ump_info.text.split('Umpire:')[-1].split('
')[0].strip()
                if home_team in teams and away_team in teams:
                    return ump_name
        return None
    except Exception as e:
        print("Failed to get scheduled umpire:", e)
        return None
    except Exception as e:
        print("Failed to get scheduled umpire:", e)
        return None
    except Exception as e:
        print("Failed to get scheduled umpire:", e)
        return None

# --- Simulated Odds Aggregator (Stub until real scraper is added) ---
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
        return {},
        'Kevin Gausman': {'DK': 6.5, 'FD': 6.0, 'B365': 6.5},
        'Zac Gallen': {'DK': 5.5, 'FD': 5.5, 'B365': 6.0},
        # Add more as needed
    }

# --- Main Pitcher Data Builder ---
def get_pitchers_by_date(selected_date):
    pitchers = []
    pitcher_stats = get_fangraphs_pitcher_stats()
    team_batting = get_team_batting_stats()
    umpire_k_factors = get_umpire_k_factors()
    odds_data = get_strikeout_odds()

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={selected_date}&hydrate=probablePitcher"
    response = requests.get(url)
    data = response.json()

    try:
        games = data['dates'][0]['games']
        for game in games:
            teams = game.get('teams', {})
            home_team = teams.get('home', {}).get('team', {}).get('name', 'Home')
            away_team = teams.get('away', {}).get('team', {}).get('name', 'Away')
            umpire_name = get_scheduled_umpire(home_team, away_team)
            k_factor = umpire_k_factors.get(umpire_name, 1.00)

            for side in ['home', 'away']:
                pitcher_info = teams.get(side, {}).get('probablePitcher')
                if pitcher_info:
                    pitcher_name = pitcher_info['fullName']
                    pitcher_team = home_team if side == 'home' else away_team
                    opponent_team = away_team if side == 'home' else home_team
                    matchup = f"vs {opponent_team}" if side == 'home' else f"@ {opponent_team}"

                    stats_row = pitcher_stats[pitcher_stats['Pitcher'].str.contains(pitcher_name.split()[-1], case=False)]
                    k9 = stats_row['Avg_K_9'].values[0] if not stats_row.empty else np.nan
                    ip = stats_row['Innings_Pitched'].values[0] if not stats_row.empty else np.nan

                    opp_stats = team_batting.get(opponent_team, {})
                    k_rate = opp_stats.get('K%', np.nan)
                    ba = opp_stats.get('BA', np.nan)
                    obp = opp_stats.get('OBP', np.nan)
                    wrc = opp_stats.get('wRC+', np.nan)

                    odds = odds_data.get(pitcher_name, {'DK': np.nan, 'FD': np.nan, 'B365': np.nan})

                    pitcher_data = {
                        'Pitcher': pitcher_name,
                        'Team': pitcher_team,
                        'Matchup': matchup,
                        'Avg_K_9': k9,
                        'Innings_Pitched': ip,
                        'Opponent_K_Rate': k_rate,
                        'Opponent_BA': ba,
                        'Opponent_OBP': obp,
                        'Opponent_WRC_Plus': wrc,
                        'Opponent_vs_Handedness_KRate': k_rate,
                        'Umpire_K_Factor': k_factor,
                        'DK_Line': odds['DK'],
                        'FD_Line': odds['FD'],
                        'B365_Line': odds['B365']
                    }
                    pitchers.append(pitcher_data)
    except Exception as e:
        print("Error parsing MLB API:", e)

    return pd.DataFrame(pitchers)
