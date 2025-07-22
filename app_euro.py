import streamlit as st
import pandas as pd
from statsbombpy import sb
import pandas as pd
import numpy as np
#from mplsoccer import Pitch
from mplsoccer import VerticalPitch,Pitch
from mplsoccer.pitch import Pitch
from highlight_text import ax_text, fig_text
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import math
import plotly.graph_objects as go
from mplsoccer import VerticalPitch
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects


st.title('Euro 2024')

#call the statsbombpy API to get a list of matches for a given competition
#Euro 2024 competition id = 55, season id = 282
euro_2024_matches = sb.matches(competition_id=55, season_id=282)

#print the first 5 matches listed
euro_2024_matches.head(5)

#concat home and away teams to keep the unique teams that participated to the tournament
home_team = euro_2024_matches['home_team']
away_team = euro_2024_matches['away_team']
teams = pd.concat([home_team,away_team])
teams = pd.DataFrame(teams.drop_duplicates()).reset_index(drop=True)
teams.columns = ['Team']

# give teams colors for the visuals
team_colors = pd.read_excel('Team Colors.xlsx')
teams = teams.merge(team_colors, how='inner', on='Team')

# take all matches in one dataframe
matches = euro_2024_matches.filter(['match_id','match_date','home_team','away_team','competition_stage','home_score','away_score','period'])
matches['match'] = matches['competition_stage'] + ' ' + matches['home_team'] + ' - ' + matches['away_team'] 

matches['match_date'] = pd.to_datetime(matches['match_date'], infer_datetime_format=True)
matches = matches.sort_values(by='match_date', ascending=False)

def take_matchid(df, value):
    match = df[df['match'] == value]
    if not match.empty:
        return match['match_id'].iloc[0]
    else:
        return None  # or handle it as you need
    

def teams_selected(df,value):
    df_details = df[df.match == value]
    return df_details


# filled the dataframe with the calculations of the stats
def add_to_dataframe(data_frame,series,column):
    # Convert series to DataFrames
    df_group_by_1 = series.reset_index()
    df_group_by_1.columns = ['team', column]
    df_group_by_1.set_index('team', inplace=True)
    # Update the main DataFrame with new columns
    return data_frame.join(df_group_by_1)

# Dropdown for selecting a match
match_selected = st.selectbox("Select a match:", matches['match'])
#call the statsbombpy events API to bring in the event data for the match
match = sb.events(match_id=take_matchid(matches,match_selected))
# remove penalty shotout if exists
match = match[match.period != 5]

# general inf of the match
match_details = teams_selected(matches,match_selected)
st.title(str(match_details['home_team'].iloc[0]) + ' ' + str(match_details['home_score'].iloc[0])+ ' : ' + str(match_details['away_score'].iloc[0])+ ' ' + str(match_details['away_team'].iloc[0]))
# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Match Overview", "XGoals","Shots", "Passes"])

# First tab: Match Analysis
with tab1:
    st.header("Match Overview")
    
    # Dropdown for selecting a match
   # match_selected = st.selectbox("Select a match:", matches['match'])

    #call the statsbombpy events API to bring in the event data for the match
    #match = sb.events(match_id=take_matchid(matches,match_selected))
    # remove penalty shotout if exists
    #match = match[match.period != 5]

    # general inf of the match
    #match_details = teams_selected(matches,match_selected)

    # Create a DataFrame with the correct index
    index = [str(match_details['home_team'].iloc[0]), str(match_details['away_team'].iloc[0])]
    stats = pd.DataFrame(index=index)

    total_possessions = match['possession_team'].count()
    # Calculate possession counts for each team
    possession_counts = match.groupby('team')['possession_team'].count()
    # Calculate possession percentage for each team
    possession_percentage = round((possession_counts / total_possessions) * 100,2)
    stats = add_to_dataframe(stats,possession_percentage,'Ball Possession')
    

    # Xgoals
    xgoals = round(match.groupby('team')['shot_statsbomb_xg'].sum(),2)
    stats = add_to_dataframe(stats,xgoals,'XGoals') 

    # Total Shots
    shots = match[match['shot_outcome'].isnull()==False].reset_index()
    shots = shots.groupby('team')['shot_outcome'].count()
    stats = add_to_dataframe(stats,shots,'Total Shots')

    # Shots On Target
    shots_ = match[match['shot_outcome'].isnull()==False].reset_index()
    shots_ontarget = shots_[shots_['shot_outcome'].isin(['Saved', 'Goal'])]
    shots_ontarget = shots_ontarget.groupby('team')['shot_outcome'].count()
    stats = add_to_dataframe(stats,shots_ontarget,'Shots On Target')

    # Shots Off Target
    shots_ = match[match['shot_outcome'].isnull()==False].reset_index()
    shots_offtarget = shots_[~shots_['shot_outcome'].isin(['Saved', 'Goal','Blocked'])]
    shots_offtarget = shots_offtarget.groupby('team')['shot_outcome'].count()
    stats = add_to_dataframe(stats,shots_offtarget,'Shots Off Target')

    # Blocked Shots
    shots_ = match[match['shot_outcome'].isnull()==False].reset_index()
    shots_blocked = shots_[shots_['shot_outcome'].isin(['Blocked'])]
    shots_blocked = shots_blocked.groupby('team')['shot_outcome'].count()
    stats = add_to_dataframe(stats,shots_blocked,'Blocked Shots')

    # Saves
    # For the saves you must swap the groupby because we take th info from th shots and as Team has the team which made the shot and not the save
    # Get the saved shots for each team
    saves = match[match['shot_outcome'] == 'Saved']
    saved_shots = saves.groupby('team')['shot_outcome'].count()

    # Create a new Series for swapped saves
    home_team = str(match_details['home_team'].values[0])  # Use .values[0] to get the first value
    away_team = str(match_details['away_team'].values[0])

    if len(saved_shots) == 2:
        # If both teams have saved shots, swap their values
        swapped_saves = pd.Series([saved_shots.iloc[1], saved_shots.iloc[0]], index=saved_shots.index)
    elif len(saved_shots) == 1:
        # If only one team has saves, assign it to home team and 0 to the other team
        if home_team in saved_shots.index:
            swapped_saves = pd.Series([saved_shots.iloc[0], 0], index=[home_team, away_team])
        else:
            swapped_saves = pd.Series([0, saved_shots.iloc[0]], index=[home_team, away_team])

    # Add to the dataframe
    stats = add_to_dataframe(stats, swapped_saves, 'Goalkeeper Saves')

    # Corners
    corners = match[(match['play_pattern'] == 'From Corner') & (match['pass_type'] == 'Corner')]
    corners = corners.groupby('team')['play_pattern'].count()
    stats = add_to_dataframe(stats,corners,'Corners')

    # Fouls
    fouls = match[match['type'] == 'Foul Committed']
    fouls = fouls.groupby('team')['type'].count()
    stats = add_to_dataframe(stats,fouls,'Fouls')

    # Tackles
    tackle = match[(match['duel_type'] == 'Tackle')]
    tackle = tackle.groupby('team')['type'].count()
    stats = add_to_dataframe(stats,tackle,'Tackles')

    # Passes
    passes = match[match['type'] == 'Pass']
    passes = passes.groupby('team')['type'].count()
    stats = add_to_dataframe(stats,passes,'Passes')

    # Passes Completed
    passes_complete = match[match['type'] == 'Pass']
    passes_complete = passes_complete[passes_complete['pass_outcome'].isnull()]
    passes_complete = passes_complete.groupby('team')['type'].count()
    stats = add_to_dataframe(stats,passes_complete,'Passes Completed')

    # Free Kicks
    free_kick = match[(match['pass_type'] == 'Free Kick') | (match['shot_type'] == 'Free Kick')]
    free_kick = free_kick.groupby('team')['type'].count()
    stats = add_to_dataframe(stats,free_kick,'Free Kicks')

    # Yellow Cards
    #yellow_card = match[(match['foul_committed_card'] == 'Yellow Card')]
    #yellow_card = yellow_card.groupby('team')['type'].count()
    #stats = add_to_dataframe(stats,yellow_card,'Yellow Cards')
#
    ## Red Cards
    #red_card = match[(match['foul_committed_card'].isin(['Red Card','Second Yellow']))]
    #red_card = red_card.groupby('team')['type'].count()
    #stats = add_to_dataframe(stats,red_card,'Red Cards')


    # Reorder the dataframe and make the columns rows
    # Transpose the DataFrame
    stats_transposed = stats.T

    # Reformat data for the given code
    transformed_data = {
        'Stat': stats_transposed.index,
        str(match_details['home_team'].iloc[0]): stats_transposed[str(match_details['home_team'].iloc[0])],
        str(match_details['away_team'].iloc[0]): stats_transposed[str(match_details['away_team'].iloc[0])]
    }

    # Create DataFrame for visualization
    df_transformed = pd.DataFrame(transformed_data)
    df_transformed = df_transformed.reset_index(drop=True)
    df_transformed = df_transformed.fillna(0)

    home_team = str(match_details['home_team'].iloc[0])
    home_color = teams[teams['Team'] == home_team]['First_Color'].values[0]
    away_team = str(match_details['away_team'].iloc[0])
    away_color = teams[teams['Team'] == away_team]['Second_Color'].values[0]


    #st.title(str(match_details['home_team'].iloc[0]) + ' ' + str(match_details['home_score'].iloc[0])+ ' : ' + str(match_details['away_score'].iloc[0])+ ' ' + str(match_details['away_team'].iloc[0]))

    #set index to 0
    match_details = match_details.reset_index(drop=True)
   

    
    categories = df_transformed['Stat'].to_list()
    home = df_transformed[str(match_details['home_team'][0])].to_list()
    away = df_transformed[str(match_details['away_team'][0])].to_list()


    # Normalize the values (convert to percentages)
    home_total = np.sum(home)
    away_total = np.sum(away)

    # Apply logarithmic scaling for bar size
    home_log_scaled = [np.log1p(x) / np.log1p(home_total) * 100 for x in home]
    away_log_scaled = [np.log1p(x) / np.log1p(away_total) * 100 for x in away]

    # Function to format values (float rounded to 2 decimals for 'XGoals', int for other categories)
    def format_value(value, category):
        if category == 'XGoals':
            return f'{value:.2f}'
        else:
            return f'{int(value)}'

    # Create the funnel chart
    fig = go.Figure()

    # Home trace
    fig.add_trace(go.Funnel(
        name=str(match_details['home_team'][0]),
        y=categories,
        x=home_log_scaled,  # Use log-scaled values for visualization
        text=[format_value(value, category) for value, category in zip(home, categories)],  # Format based on category
        textinfo='text',
        marker=dict(color=home_color)
    ))

    # Away trace
    fig.add_trace(go.Funnel(
        name=str(match_details['away_team'][0]),
        y=categories,
        x=away_log_scaled,  # Use log-scaled values for visualization
        text=[format_value(value, category) for value, category in zip(away, categories)],  # Format based on category
        textinfo='text',
        marker=dict(color=away_color)
    ))

    # Customize layout
    fig.update_layout(
        title="Match Overview",
        height=1000  # Adjust height
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)



# Placeholder for other tab
with tab2:
   

    # XGoals Analysis 

    xg = match[match.type == 'Shot']
    xg = xg[['period', 'minute', 'shot_statsbomb_xg', 'team', 'player', 'shot_outcome']]
    # remove penalty shout out if existed at the end of the game
    xg =  xg[xg.period != 5]
    xg.rename(columns = {'shot_statsbomb_xg':'xG', 'shot_outcome':'result'}, inplace = True)
    xg.sort_values(by='team', inplace=True)
    xg.head()

    hteam = str(match_details['home_team'].iloc[0])
    ateam = str(match_details['away_team'].iloc[0])

    #Cumulative Sum xG home team
    h_xg = xg[xg['team'] == hteam]
    h_xg = h_xg.sort_values(by='minute').copy()  # Use copy() to avoid view assignment
    h_xg.loc[:, 'h_cum'] = h_xg['xG'].cumsum()

    #Cumulative Sum xG away team
    a_xg = xg[xg['team'] == ateam]
    a_xg = a_xg.sort_values(by='minute').copy()
    a_xg.loc[:,'a_cum'] = a_xg['xG'].cumsum()

    h_goal = h_xg[h_xg['result'].str.contains("Goal")]
    h_goal["scorechart"] = h_goal["minute"].astype(str) + "'" + " " +h_goal["player"]
    a_goal = a_xg[a_xg['result'].str.contains("Goal")]
    a_goal["scorechart"] = a_goal["minute"].astype(str) + "'" + " " +a_goal["player"]

    #Total xG
    a_total = round(a_xg['xG'].sum(),2).astype(str)
    h_total = round(h_xg['xG'].sum(),2).astype(str)

    #scores
    #h_goal_scorers = h_goal[['player','minute','team']]
    #a_goal_scorers = a_goal[['player','minute','team']]
#
    #goal_scorers = pd.concat([h_goal_scorers, a_goal_scorers], axis=0)
    #goal_scorers.columns = ['Scorers','Minute','Team']
    #goal_scorers = goal_scorers.sort_values(by=['Minute'])
    goal_scorers = match[(match['type'] == 'Own Goal Against') | (match['shot_outcome'] == 'Goal')]
    
    goal_scorers['goal_type'] = np.where(
    goal_scorers['type'] == 'Own Goal Against', 'Own Goal', 
    np.where(goal_scorers['shot_outcome'] == 'Goal', 'Goal', None))

    goal_scorers = goal_scorers[['player','minute','team','goal_type']]
    goal_scorers.columns = ['Scorers','Minute','Team','Goal Type']

    goal_scorers = goal_scorers.sort_values('Minute')
    st.write('Score Board')
   
    st.markdown(goal_scorers.style.hide(axis="index").to_html(), unsafe_allow_html=True)

    st.title('XGoals Analysis')

    # Create the figure
    fig = go.Figure()

    # Add line plots for the expected goals
    fig.add_trace(go.Scatter(
        x=h_xg['minute'], y=h_xg['h_cum'], mode='lines', 
        line_shape='hv', line=dict(color=home_color, width=2),
        name=f"{hteam} ({h_total} xG)"
    ))
    fig.add_trace(go.Scatter(
        x=a_xg['minute'], y=a_xg['a_cum'], mode='lines', 
        line_shape='hv', line=dict(color=away_color, width=2),
        name=f"{ateam} ({a_total} xG)"
    ))

    # Add scatter plot for goals
    fig.add_trace(go.Scatter(
        x=h_goal['minute'], y=h_goal['h_cum'], mode='markers', 
        marker=dict(color=home_color, size=12, symbol='circle'), 
        name=f"{hteam} Goals"
    ))
    fig.add_trace(go.Scatter(
        x=a_goal['minute'], y=a_goal['a_cum'], mode='markers', 
        marker=dict(color=away_color, size=12, symbol='circle'), 
        name=f"{ateam} Goals"
    ))



    # Add annotations for goals
    for j, txt in h_goal['scorechart'].items():
        fig.add_annotation(x=h_goal['minute'][j], y=h_goal['h_cum'][j],
                           text=txt, showarrow=True, arrowhead=2, ax=-90, ay=50,
                           arrowcolor='white')

    for i, txt in a_goal['scorechart'].items():
        fig.add_annotation(x=a_goal['minute'][i], y=a_goal['a_cum'][i],
                           text=txt, showarrow=True, arrowhead=2, ax=-90, ay=50,
                           arrowcolor='white')


    # Customize the chart layout
    fig.update_layout(
        title={
            'text': f"{hteam} - {ateam} ({str(match_details['home_score'].iloc[0])} - {str(match_details['away_score'].iloc[0])})",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': 'white'}
        },
        annotations=[dict(text= str(match_details['competition_stage'].iloc[0]) , xref='paper', x=0.125, y=0.92, showarrow=False, font=dict(size=18))],
        xaxis_title='Minutes',
        yaxis_title='Expected Goals (xG)',
        xaxis=dict(tickvals=[0, 15, 30, 45, 60, 75, 90]),
        yaxis=dict(tickvals=[0, 0.5, 1, 1.5, 2, 2.5, 3]),
        #legend_title='Total Expected Goals (xG)',
        #legend=dict(x=0.02, y=0.98),
        template='plotly_white'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


    #Xgoals Pitch
    shots_xg = match[['location', 'minute', 'player', 'team', 'shot_outcome', 'shot_statsbomb_xg', 'shot_technique', 'shot_type','period']]
    # remove if penalyty shotout exists in th end
    shots_xg = shots_xg[shots_xg.period != 5]
    shots_xg = shots_xg[shots_xg['shot_outcome'].isnull()==False].reset_index()
    shots_Loc = shots_xg['location']
    shots_Loc = pd.DataFrame(shots_Loc.to_list(), columns=['x', 'y'])
    shots_xg['x'] = shots_Loc['x']
    shots_xg['y'] = shots_Loc['y']
    shots_xg = shots_xg.drop(['index','location'], axis=1)
    shots_xg_home = shots_xg[shots_xg['team'] == str(match_details['home_team'].iloc[0])].reset_index()
    shots_xg_away = shots_xg[shots_xg['team'] == str(match_details['away_team'].iloc[0])].reset_index()

    # Extract the expected goals and the time data for both the teams
    home_xg = shots_xg_home['shot_statsbomb_xg'].tolist()
    home_minute = shots_xg_home['minute'].tolist()

    away_xg = shots_xg_away['shot_statsbomb_xg'].tolist()
    away_minute = shots_xg_away['minute'].tolist()

    # generate the cumulative expected goals values
    home_xg_cumu = np.cumsum(home_xg)
    away_xg_cumu = np.cumsum(away_xg)




    # Create two columns for side-by-side layout in Streamlit
    col1, col2 = st.columns(2)

    # Home team plot
    with col1:
        # Set up the pitch and figure for the home team
        pitch_color = '#0E1117'  # Same color as the pitch
        line_color = '#c7d5cc'   # Pitch lines color
        pitch = VerticalPitch(
            pitch_type='statsbomb', 
            half=True, 
            pitch_color=pitch_color, 
            pad_bottom=.5, 
            line_color='white',
            linewidth=.75,
            axis=True, label=True
        )

        # Create the figure and axis
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Draw the pitch for the home team
        pitch.draw(ax=ax1)

        # Loop over each shot for the home team to plot the bubbles with the required conditions
        for i in range(len(shots_xg_home)):
            x, y = shots_xg_home.x[i], shots_xg_home.y[i]
            outcome = shots_xg_home.shot_outcome[i]
            size = 400 if outcome == 'Goal' else 1000 * shots_xg_home.shot_statsbomb_xg[i]

            if outcome == 'Goal':
                # For Goals: Full marker using football marker
                pitch.scatter(x, y, marker='football', s=size, ax=ax1)
            else:
                color = '#FF4B4B'

                # Plot without filling the inside, only outline
                pitch.scatter(x, y, edgecolors=color, facecolors='none', linewidth=2, s=size, ax=ax1)

        # Set title for the home team plot
        ax1.set_title(f"{str(match_details['home_team'].iloc[0])} xG: {str(round(home_xg_cumu[-1], 2))}", 
                      size=20, color='white')
        ax1.set_axis_off()

        # Set the background color of the figure to match the pitch
        fig.patch.set_facecolor(pitch_color)

        # Display the home team plot in Streamlit
        st.pyplot(fig)


    # Away team plot
    with col2:
        # Set up the pitch and figure for the away team
        fig, ax2 = plt.subplots(figsize=(10, 6))

        # Draw the pitch for the away team
        pitch.draw(ax=ax2)

        # Loop over each shot for the away team to plot the bubbles with the required conditions
        for i in range(len(shots_xg_away)):
            x, y = shots_xg_away.x[i], shots_xg_away.y[i]
            outcome = shots_xg_away.shot_outcome[i]
            size = 400 if outcome == 'Goal' else 1000 * shots_xg_away.shot_statsbomb_xg[i]

            if outcome == 'Goal':
                # For Goals: Full marker using football marker
                pitch.scatter(x, y, marker='football', s=size, ax=ax2)
            else:
                color = '#FF4B4B'

                # Plot without filling the inside, only outline
                pitch.scatter(x, y, edgecolors=color, facecolors='none', linewidth=2, s=size, ax=ax2)

        # Set title for the away team plot
        ax2.set_title(f"{str(match_details['away_team'].iloc[0])} xG: {str(round(away_xg_cumu[-1], 2))}", 
                      size=20, color='white')
        ax2.set_axis_off()

        # Set the background color of the figure to match the pitch
        fig.patch.set_facecolor(pitch_color)

        # Display the away team plot in Streamlit
        st.pyplot(fig)
    
# Placeholder for other tab
with tab3: 

    shots = match[['team', 'type', 'minute', 'location', 'shot_end_location', 'shot_outcome', 'player']]

    shots = shots[shots['type'].isin(['Shot'])]

    shots_home = shots[shots['team'] == str(match_details['home_team'].iloc[0])].reset_index()
    shots_away = shots[shots['team'] == str(match_details['away_team'].iloc[0])].reset_index()
    shots_Loc_home = shots_home['location']
    shots_Loc_home = pd.DataFrame(shots_Loc_home.to_list(), columns=['x', 'y'])

    shots_Loc_away = shots_away['location']
    shots_Loc_away = pd.DataFrame(shots_Loc_away.to_list(), columns=['x', 'y'])


    num_shots = len(shots_home)

    # Display the team name and the number of shots
    st.title(f"**{str(match_details['home_team'].iloc[0])} Shots: {num_shots}**")

    # Calculate the number of shots on target and off target
    on_target = shots_home[shots_home['shot_outcome'].isin(['Goal', 'Saved'])].shape[0]
    off_target = shots_home[~shots_home['shot_outcome'].isin(['Goal', 'Saved'])].shape[0]

    # Pie chart for shots distribution (On Target vs Off Target)
    labels = ['On Target', 'Off Target']
    values = [on_target, off_target]
    colors = ['#FF4B4B', '#FAFAFA']

    # Create a donut chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=colors))])

    # Update layout for donut chart
    fig.update_layout(
        title_text="Shots On Target vs Off Target",
        annotations=[dict(text=str(match_details['home_team'].iloc[0]), x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)





# Set up your dataframes shots_home and match_details beforehand

    # Streamlit title
    st.title(f"{str(match_details['home_team'].iloc[0])} Shots Visualization")

    # Create the figure and axes with the desired figsize
    fig, ax = plt.subplots(figsize=(13.5, 8), constrained_layout=True)

    # Set up the pitch (without orientation) with a background color that matches the pitch
    pitch_color = '#0E1117'  # Same color as the pitch
    line_color = '#c7d5cc'   # Pitch lines color
    pitch = VerticalPitch(
        pitch_type='statsbomb', 
        half=True, 
        pitch_color='#0E1117', 
        pad_bottom=.5, 
        line_color='white',
        linewidth=.75,
        axis=True, label=True
    )

    # Draw the pitch on the axes
    pitch.draw(ax=ax)

    # Plot shots with different outcomes (Goal, Blocked, Saved, Other)
    for i in range(len(shots_home)):
        if shots_home.shot_outcome[i] == 'Goal':
            pitch.arrows(shots_home.location[i][0], shots_home.location[i][1], shots_home.shot_end_location[i][0], shots_home.shot_end_location[i][1], ax=ax, color='green', width=3)
            pitch.scatter(shots_home.location[i][0], shots_home.location[i][1], ax=ax, color='green', alpha=1)
        elif shots_home.shot_outcome[i] in ['Blocked', 'Saved']:
            pitch.arrows(shots_home.location[i][0], shots_home.location[i][1], shots_home.shot_end_location[i][0], shots_home.shot_end_location[i][1], ax=ax, color='red', width=3)
            pitch.scatter(shots_home.location[i][0], shots_home.location[i][1], ax=ax, color='red', alpha=1)
        else:
            pitch.arrows(shots_home.location[i][0], shots_home.location[i][1], shots_home.shot_end_location[i][0], shots_home.shot_end_location[i][1], ax=ax, color='orange', width=3)
            pitch.scatter(shots_home.location[i][0], shots_home.location[i][1], ax=ax, color='orange', alpha=1)

    # Add a title
    ax.set_title(f"{str(match_details['home_team'].iloc[0])} Shots : {str(len(shots_home))}", 
                 size=20, color='white')

    # Remove tick numbers around the pitch
    ax.set_xticks([])
    ax.set_yticks([])

    # Set background color for the figure
    fig.patch.set_facecolor(pitch_color)

    # Add a legend
    goal_patch = mpatches.Patch(color='green', label='Goal')
    blocked_patch = mpatches.Patch(color='red', label='Blocked/Saved')
    other_patch = mpatches.Patch(color='orange', label='Out')
    ax.legend(handles=[goal_patch, blocked_patch, other_patch], loc='upper right', fontsize=12, facecolor='white', edgecolor='black')

    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(fig)

    num_shots = len(shots_away)

    # Display the team name and the number of shots
    st.title(f"**{str(match_details['away_team'].iloc[0])} Shots: {num_shots}**")

    # Calculate the number of shots on target and off target
    on_target = shots_away[shots_away['shot_outcome'].isin(['Goal', 'Saved'])].shape[0]
    off_target = shots_away[~shots_away['shot_outcome'].isin(['Goal', 'Saved'])].shape[0]

    # Pie chart for shots distribution (On Target vs Off Target)
    labels = ['On Target', 'Off Target']
    values = [on_target, off_target]
    colors = ['#FF4B4B', '#FAFAFA']

    # Create a donut chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=colors))])

    # Update layout for donut chart
    fig.update_layout(
        title_text="Shots On Target vs Off Target",
        annotations=[dict(text=str(match_details['away_team'].iloc[0]), x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)



    # Streamlit title
    st.title(f"{str(match_details['away_team'].iloc[0])} Shots Visualization")

    # Create the figure and axes with the desired figsize
    fig, ax = plt.subplots(figsize=(13.5, 8), constrained_layout=True)

    # Set up the pitch (without orientation) with a background color that matches the pitch
    pitch_color = '#0E1117'  # Same color as the pitch
    line_color = '#c7d5cc'   # Pitch lines color
    pitch = VerticalPitch(
        pitch_type='statsbomb', 
        half=True, 
        pitch_color='#0E1117', 
        pad_bottom=.5, 
        line_color='white',
        linewidth=.75,
        axis=True, label=True
    )

    # Draw the pitch on the axes
    pitch.draw(ax=ax)

    # Plot shots with different outcomes (Goal, Blocked, Saved, Other)
    for i in range(len(shots_away)):
        if shots_away.shot_outcome[i] == 'Goal':
            pitch.arrows(shots_away.location[i][0], shots_away.location[i][1], shots_away.shot_end_location[i][0], shots_away.shot_end_location[i][1], ax=ax, color='green', width=3)
            pitch.scatter(shots_away.location[i][0], shots_away.location[i][1], ax=ax, color='green', alpha=1)
        elif shots_away.shot_outcome[i] in ['Blocked', 'Saved']:
            pitch.arrows(shots_away.location[i][0], shots_away.location[i][1], shots_away.shot_end_location[i][0], shots_away.shot_end_location[i][1], ax=ax, color='red', width=3)
            pitch.scatter(shots_away.location[i][0], shots_away.location[i][1], ax=ax, color='red', alpha=1)
        else:
            pitch.arrows(shots_away.location[i][0], shots_away.location[i][1], shots_away.shot_end_location[i][0], shots_away.shot_end_location[i][1], ax=ax, color='orange', width=3)
            pitch.scatter(shots_away.location[i][0], shots_away.location[i][1], ax=ax, color='orange', alpha=1)

    # Add a title
    ax.set_title(f"{str(match_details['away_team'].iloc[0])} Shots : {str(len(shots_away))}", 
                 size=20, color='white')

    # Remove tick numbers around the pitch
    ax.set_xticks([])
    ax.set_yticks([])

    # Set background color for the figure
    fig.patch.set_facecolor(pitch_color)

    # Add a legend
    goal_patch = mpatches.Patch(color='green', label='Goal')
    blocked_patch = mpatches.Patch(color='red', label='Blocked/Saved')
    other_patch = mpatches.Patch(color='orange', label='Out')
    ax.legend(handles=[goal_patch, blocked_patch, other_patch], loc='upper right', fontsize=12, facecolor='white', edgecolor='black')

    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(fig)



with tab4:
    event_pass = match[['minute', 'second', 'team', 'location', 'period', 'type', 'pass_outcome', 'player', 'position', 'pass_end_location']]
    event_pass_home=event_pass[event_pass['team']==str(match_details['home_team'].iloc[0])]
    # Add the passer and the recipient columns
    event_pass_home['passer'] = event_pass_home['player']
    event_pass_home['recipient'] = event_pass_home['player'].shift(-1)
    passes_home = event_pass_home[event_pass_home['type'] == 'Pass']
    successful_home = passes_home[passes_home['pass_outcome'].isnull()]
    # Pass network before the first substitution takes place
    subs_home = event_pass_home[event_pass_home['type']=='Substitution']
    # Time when the first substituion took place
    first_sub_home_minute = subs_home['minute'].min()
    first_sub_home_minute_df = subs_home[subs_home['minute'] == first_sub_home_minute]
    first_sub_home_second = first_sub_home_minute_df['second'].min()
    # Filter oute the data for generating pass network before the first substitution takes place
    successful_home = successful_home[(successful_home['minute']<=first_sub_home_minute-1) ]
    pass_loc_home = successful_home['location']
    pass_loc_home = pd.DataFrame(pass_loc_home.to_list(), columns=['x', 'y'])
    pass_end_loc_home = successful_home['pass_end_location']
    pass_end_loc_home = pd.DataFrame(pass_end_loc_home.to_list(), columns=['end_x', 'end_y'])
    successful_home=successful_home.reset_index()
    successful_home['x'] = pass_loc_home['x']
    successful_home['y'] = pass_loc_home['y']
    successful_home['end_x'] = pass_end_loc_home['end_x']
    successful_home['end_y'] = pass_end_loc_home['end_y']
    del successful_home['location']
    del successful_home['pass_end_location']

    successful_home['pass_outcome'] = 'successful'

    # find the average locations of the passer
    avg_loc_home = successful_home.groupby('passer').agg({'x':['mean'], 'y': ['mean', 'count']})

    avg_loc_home.columns=['x', 'y', 'count']
    # Number of passes between each player
    pass_bet_home = successful_home.groupby(['passer', 'recipient']).index.count().reset_index()

    pass_bet_home.rename({'index':'pass_count'}, axis='columns', inplace=True)
    pass_bet_home = pass_bet_home.merge(avg_loc_home, left_on = 'passer', right_index=True)
    pass_bet_home = pass_bet_home.merge(avg_loc_home, left_on = 'recipient', right_index=True, suffixes=['', '_end'])


    num_passes = len(passes_home)

    # Display the team name and the number of shots
    st.title(f"**{str(match_details['home_team'].iloc[0])} Passes: {num_passes}**")

  

    # Display the team name and the number of shots
    #st.write(f"**{team_name} Shots: {num_shots}**")

    # Calculate the number of shots on target and off target
    succesfull = passes_home[passes_home['pass_outcome'].isna()].shape[0]
    not_succesfull = passes_home[~passes_home['pass_outcome'].isna()].shape[0]

    # Pie chart for shots distribution (On Target vs Off Target)
    labels = ['Succesfull', 'Not Succesfull']
    values = [succesfull, not_succesfull]
    colors = ['#FF4B4B', '#FAFAFA']

    # Create a donut chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=colors))])

    # Update layout for donut chart
    fig.update_layout(
        title_text="Passes",
        annotations=[dict(text=str(match_details['away_team'].iloc[0]), x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True
    )

    st.plotly_chart(fig)


    # Create the pitch
    pitch_color = '#0E1117'  # Same color as the pitch
    line_color = '#c7d5cc'   # Pitch lines color
    pitch = Pitch(pitch_type='statsbomb', pitch_color=pitch_color, line_color=line_color)

    # Create a matplotlib figure and axes
    fig, ax = pitch.draw(figsize=(13.5, 8))

    # Plot the arrows (representing passes)
    arrows = pitch.arrows(pass_bet_home.x, pass_bet_home.y, pass_bet_home.x_end, pass_bet_home.y_end, ax=ax,
                          width=5, headwidth=3, color='white', zorder=1, alpha=0.5)

    # Plot the nodes (representing average player locations)
    nodes = pitch.scatter(avg_loc_home.x, avg_loc_home.y, s=400, color=home_color, edgecolors='black', linewidth=2.5,
                          alpha=1, zorder=1, ax=ax)

    # Set the background color of the figure to match the pitch
    fig.patch.set_facecolor(pitch_color)

    # Set the title
    ax.set_title(f"Pass Network of: {match_details['home_team'].iloc[0]}", color='white', size=16)

    # Display the figure in Streamlit
    st.pyplot(fig)

    match_pass_perc = passes_home['location']
    match_pass_perc = pd.DataFrame(match_pass_perc.to_list(), columns=['x', 'y'])
    
    total = len(match_pass_perc)
    df_pass_perc = pd.DataFrame(columns=['Team','Def 3rd','Mid 3rd','Att 3rd'])
    
    new_row = pd.DataFrame({
        'Team': [str(match_details['home_team'].iloc[0])],
        'Def 3rd': [len(match_pass_perc[match_pass_perc.x <= 40])],
        'Mid 3rd': [len(match_pass_perc[(match_pass_perc['x'] > 40) & (match_pass_perc['x'] < 80)])],
        'Att 3rd': [len(match_pass_perc[match_pass_perc.x >= 80])],
        'Total': [total]
    })
    
    # Calculate the percentage of passes in each third
    new_row['Def 3rd (%)'] = (new_row['Def 3rd'] / total) * 100
    new_row['Mid 3rd (%)'] = (new_row['Mid 3rd'] / total) * 100
    new_row['Att 3rd (%)'] = (new_row['Att 3rd'] / total) * 100
    
    # Concatenate the new row to the existing DataFrame
    df_pass_perc = pd.concat([df_pass_perc, new_row], ignore_index=True)
    
    # path effects
    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]
    
    vmin = df_pass_perc[['Def 3rd (%)', 'Mid 3rd (%)', 'Att 3rd (%)']].values.min()
    vmax = df_pass_perc[['Def 3rd (%)', 'Mid 3rd (%)', 'Att 3rd (%)']].values.max()
    
    # setup a mplsoccer pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0E1117', line_zorder=2, line_color='#c7d5cc')
    bin_statistic = pitch.bin_statistic([0], [0], statistic='count', bins=(3, 1))
    
    fig, ax = pitch.draw(figsize=(16, 11),constrained_layout=True, tight_layout=False)
    fig.set_facecolor('#0E1117')
    
    # path effects
    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]
    
    
            
    # fill in the bin statistics from df and plot the heatmap
    bin_statistic['statistic'] = df_pass_perc[['Def 3rd (%)', 'Mid 3rd (%)', 'Att 3rd (%)']].values
    heatmap = pitch.heatmap(bin_statistic, ax=ax, cmap='Purples', vmin=vmin, vmax=vmax)
    annotate = pitch.label_heatmap(bin_statistic, color='white', #fontproperties=fm.prop,
                                   path_effects=path_eff, fontsize=50, ax=ax,
                                   str_format='{0:.0f}%', ha='center', va='center')
    
    st.pyplot(fig)



    #separate start and end locations from coordinates
    match[['x', 'y']] = match['location'].apply(pd.Series)
    match[['pass_end_x', 'pass_end_y']] = match['pass_end_location'].apply(pd.Series)
    match[['carry_end_x', 'carry_end_y']] = match['carry_end_location'].apply(pd.Series)

    #create a variable for the team you want to look into
    team=str(match_details['home_team'].iloc[0])


    passes_df=match[(match.team==team)&(match.type=="Pass")&(match.x<80)&(match.pass_end_x>80)&(match.pass_outcome.isna())]

    #Visualize for a team
    pass_colour='#FF4B4B'

    #set up the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0E1117', line_zorder=2, line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(16, 11),constrained_layout=True, tight_layout=False)
    fig.set_facecolor('#D3D3D3')

    #plot the passes
    pitch.arrows(passes_df.x, passes_df.y,
    passes_df.pass_end_x, passes_df.pass_end_y, width=3,
    headwidth=8, headlength=5, color=pass_colour, ax=ax, zorder=2, label = "Pass")

    #plot the legend
    ax.legend(facecolor='#D3D3D3', handlelength=5, edgecolor='None', fontsize=20, loc='best')
    # Set the background color of the figure to match the pitch
    fig.patch.set_facecolor(pitch_color)
    #set title of viz
    ax_title = ax.set_title(f'{team} Progressions into Final 3rd {len(passes_df)} passes', fontsize=30,color='white')
    st.pyplot(fig)



    event_pass = match[['minute', 'second', 'team', 'location', 'period', 'type', 'pass_outcome', 'player', 'position', 'pass_end_location']]
    event_pass_away=event_pass[event_pass['team']==str(match_details['away_team'].iloc[0])]
    # Add the passer and the recipient columns
    event_pass_away['passer'] = event_pass_away['player']
    event_pass_away['recipient'] = event_pass_away['player'].shift(-1)
    passes_away = event_pass_away[event_pass_away['type'] == 'Pass']
    successful_away = passes_away[passes_away['pass_outcome'].isnull()]
    # Pass network before the first substitution takes place
    subs_away = event_pass_away[event_pass_away['type']=='Substitution']
    # Time when the first substituion took place
    first_sub_away_minute = subs_away['minute'].min()
    first_sub_away_minute_df = subs_away[subs_away['minute'] == first_sub_away_minute]
    first_sub_away_second = first_sub_away_minute_df['second'].min()
    # Filter oute the data for generating pass network before the first substitution takes place
    successful_away = successful_away[(successful_away['minute']<=first_sub_away_minute-1) ]
    pass_loc_away = successful_away['location']
    pass_loc_away = pd.DataFrame(pass_loc_away.to_list(), columns=['x', 'y'])
    pass_end_loc_away = successful_away['pass_end_location']
    pass_end_loc_away = pd.DataFrame(pass_end_loc_away.to_list(), columns=['end_x', 'end_y'])
    successful_away=successful_away.reset_index()
    successful_away['x'] = pass_loc_away['x']
    successful_away['y'] = pass_loc_away['y']
    successful_away['end_x'] = pass_end_loc_away['end_x']
    successful_away['end_y'] = pass_end_loc_away['end_y']
    del successful_away['location']
    del successful_away['pass_end_location']

    successful_away['pass_outcome'] = 'successful'

    # find the average locations of the passer
    avg_loc_away = successful_away.groupby('passer').agg({'x':['mean'], 'y': ['mean', 'count']})

    avg_loc_away.columns=['x', 'y', 'count']
    # Number of passes between each player
    pass_bet_away = successful_away.groupby(['passer', 'recipient']).index.count().reset_index()

    pass_bet_away.rename({'index':'pass_count'}, axis='columns', inplace=True)
    pass_bet_away = pass_bet_away.merge(avg_loc_away, left_on = 'passer', right_index=True)
    pass_bet_away = pass_bet_away.merge(avg_loc_away, left_on = 'recipient', right_index=True, suffixes=['', '_end'])


    num_passes = len(passes_away)

    # Display the team name and the number of shots
    st.title(f"**{str(match_details['away_team'].iloc[0])} Passes: {num_passes}**")

  

    # Display the team name and the number of shots
    #st.write(f"**{team_name} Shots: {num_shots}**")

    # Calculate the number of shots on target and off target
    succesfull = passes_away[passes_away['pass_outcome'].isna()].shape[0]
    not_succesfull = passes_away[~passes_away['pass_outcome'].isna()].shape[0]

    # Pie chart for shots distribution (On Target vs Off Target)
    labels = ['Succesfull', 'Not Succesfull']
    values = [succesfull, not_succesfull]
    colors = ['#FF4B4B', '#FAFAFA']

    # Create a donut chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=colors))])

    # Update layout for donut chart
    fig.update_layout(
        title_text="Passes",
        annotations=[dict(text=str(match_details['away_team'].iloc[0]), x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True
    )

    st.plotly_chart(fig)


    # Create the pitch
    pitch_color = '#0E1117'  # Same color as the pitch
    line_color = '#c7d5cc'   # Pitch lines color
    pitch = Pitch(pitch_type='statsbomb', pitch_color=pitch_color, line_color=line_color)
    
    # Create a matplotlib figure and axes
    fig, ax = pitch.draw(figsize=(13.5, 8))
    
    # Plot the arrows (representing passes)
    arrows = pitch.arrows(pass_bet_away.x, pass_bet_away.y, pass_bet_away.x_end, pass_bet_away.y_end, ax=ax,
                          width=5, headwidth=3, color='white', zorder=1, alpha=0.5)
    
    # Plot the nodes (representing average player locations)
    nodes = pitch.scatter(avg_loc_away.x, avg_loc_away.y, s=400, color=away_color, edgecolors='black', linewidth=2.5,
                          alpha=1, zorder=1, ax=ax)
    
    # Set the background color of the figure to match the pitch
    fig.patch.set_facecolor(pitch_color)
    
    # Set the title
    ax.set_title(f"Pass Network of: {match_details['away_team'].iloc[0]}", color='white', size=16)
    
    # Display the figure in Streamlit
    st.pyplot(fig)


    match_pass_perc = passes_away['location']
    match_pass_perc = pd.DataFrame(match_pass_perc.to_list(), columns=['x', 'y'])
    
    total = len(match_pass_perc)
    df_pass_perc = pd.DataFrame(columns=['Team','Def 3rd','Mid 3rd','Att 3rd'])
    
    new_row = pd.DataFrame({
        'Team': [str(match_details['home_team'].iloc[0])],
        'Def 3rd': [len(match_pass_perc[match_pass_perc.x <= 40])],
        'Mid 3rd': [len(match_pass_perc[(match_pass_perc['x'] > 40) & (match_pass_perc['x'] < 80)])],
        'Att 3rd': [len(match_pass_perc[match_pass_perc.x >= 80])],
        'Total': [total]
    })
    
    # Calculate the percentage of passes in each third
    new_row['Def 3rd (%)'] = (new_row['Def 3rd'] / total) * 100
    new_row['Mid 3rd (%)'] = (new_row['Mid 3rd'] / total) * 100
    new_row['Att 3rd (%)'] = (new_row['Att 3rd'] / total) * 100
    
    # Concatenate the new row to the existing DataFrame
    df_pass_perc = pd.concat([df_pass_perc, new_row], ignore_index=True)
    
    # path effects
    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]
    
    vmin = df_pass_perc[['Def 3rd (%)', 'Mid 3rd (%)', 'Att 3rd (%)']].values.min()
    vmax = df_pass_perc[['Def 3rd (%)', 'Mid 3rd (%)', 'Att 3rd (%)']].values.max()
    
    # setup a mplsoccer pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0E1117', line_zorder=2, line_color='#c7d5cc')
    bin_statistic = pitch.bin_statistic([0], [0], statistic='count', bins=(3, 1))
    
    fig, ax = pitch.draw(figsize=(16, 11),constrained_layout=True, tight_layout=False)
    fig.set_facecolor('#0E1117')
    
    # path effects
    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]
    
    
            
    # fill in the bin statistics from df and plot the heatmap
    bin_statistic['statistic'] = df_pass_perc[['Def 3rd (%)', 'Mid 3rd (%)', 'Att 3rd (%)']].values
    heatmap = pitch.heatmap(bin_statistic, ax=ax, cmap='Purples', vmin=vmin, vmax=vmax)
    annotate = pitch.label_heatmap(bin_statistic, color='white', #fontproperties=fm.prop,
                                   path_effects=path_eff, fontsize=50, ax=ax,
                                   str_format='{0:.0f}%', ha='center', va='center')
    
    st.pyplot(fig)






        #separate start and end locations from coordinates
    match[['x', 'y']] = match['location'].apply(pd.Series)
    match[['pass_end_x', 'pass_end_y']] = match['pass_end_location'].apply(pd.Series)
    match[['carry_end_x', 'carry_end_y']] = match['carry_end_location'].apply(pd.Series)

    #create a variable for the team you want to look into
    team=str(match_details['away_team'].iloc[0])


    passes_df=match[(match.team==team)&(match.type=="Pass")&(match.x<80)&(match.pass_end_x>80)&(match.pass_outcome.isna())]

    #Visualize for a team
    pass_colour='#FF4B4B'

    #set up the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0E1117', line_zorder=2, line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(16, 11),constrained_layout=True, tight_layout=False)
    fig.set_facecolor('#D3D3D3')

    #plot the passes
    pitch.arrows(passes_df.x, passes_df.y,
    passes_df.pass_end_x, passes_df.pass_end_y, width=3,
    headwidth=8, headlength=5, color=pass_colour, ax=ax, zorder=2, label = "Pass")

    #plot the legend
    ax.legend(facecolor='#D3D3D3', handlelength=5, edgecolor='None', fontsize=20, loc='best')
    # Set the background color of the figure to match the pitch
    fig.patch.set_facecolor(pitch_color)
    #set title of viz
    ax_title = ax.set_title(f'{team} Progressions into Final 3rd {len(passes_df)} passes', fontsize=30,color='white')
    st.pyplot(fig)