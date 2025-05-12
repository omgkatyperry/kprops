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

            # Add real DraftKings odds from scraped data (already pulled into pitchers_df earlier)
            if 'DK_Line' in pitchers_df.columns:
                pitchers_df['Edge_vs_DK'] = pitchers_df['Predicted_Ks'] - pitchers_df['DK_Line']
            else:
                pitchers_df['Edge_vs_DK'] = np.nan

            display_cols = [
                'Pitcher', 'Team', 'Matchup',
                'Avg_K_9', 'Innings_Pitched', 'Opponent_K_Rate',
                'Opponent_BA', 'Opponent_OBP', 'Opponent_WRC_Plus',
                'Predicted_Ks', 'DK_Line', 'Edge_vs_DK'
            ]

            styled_df = pitchers_df[display_cols].copy()
            styled_df[[
                'Avg_K_9', 'Innings_Pitched', 'Opponent_K_Rate',
                'Opponent_BA', 'Opponent_OBP', 'Opponent_WRC_Plus',
                'Predicted_Ks', 'DK_Line', 'Edge_vs_DK'
            ]] = styled_df[[
                'Avg_K_9', 'Innings_Pitched', 'Opponent_K_Rate',
                'Opponent_BA', 'Opponent_OBP', 'Opponent_WRC_Plus',
                'Predicted_Ks', 'DK_Line', 'Edge_vs_DK'
            ]].round(2)

            st.dataframe(
                styled_df.sort_values(by='Edge_vs_DK', ascending=False).reset_index(drop=True),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
