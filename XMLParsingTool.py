#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("Data"))
import sqlite3
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.figure_factory as ff
import ipywidgets as widgets
import math
plotly.offline.init_notebook_mode()


# In[52]:


database = "Data/database.sqlite"

conn = sqlite3.connect(database)
print(conn)


# In[53]:


pd.read_sql("SELECT * FROM sqlite_master WHERE type='table';", conn)


# In[54]:


pd.read_sql("SELECT * FROM League;", conn)


# In[24]:


#Given that we want to find out infomration of the English Premier Leage, lets get the data using SQL query
epl = pd.read_sql("""SELECT id, league_id, stage, season,date, match_api_id,
                     home_team_api_id, (SELECT team_long_name FROM Team WHERE team_api_id = home_team_api_id) home_team, 
                     away_team_api_id, (SELECT team_long_name FROM Team WHERE team_api_id = away_team_api_id) away_team,
                     home_team_goal, away_team_goal, goal, shoton, shotoff, foulcommit,
                     card, cross, corner, possession, B365H, B365D, B365A
                     FROM Match m
                     WHERE league_id = (SELECT id FROM league WHERE name = 'England Premier League')
                     ORDER BY date;
                """, conn)


# In[25]:


match_count = epl['id'].size
no_seasons = epl['season'].nunique()
print("The dataframe consists of %d rows spanning %d years of data" % (match_count, no_seasons))
epl['season'].value_counts()


# In[26]:


epl.head()


# In[27]:


epl['shoton'].iloc[0]


# In[47]:


def calculate_stats_both_teams(xml_document, home_team, away_team, card_type='y'):
    assert card_type == 'y' or card_type == 'r', "Please enter either y or r"
    tree = ET.fromstring(xml_document)
    stat_home_team = 0
    stat_away_team = 0
    
    #Dealing with card type using the root element & the card type argument
    if tree.tag == 'card':
        for child in tree.iter('value'):
            #Some xml docs have no card_type element in the tree. comment section seems to have that information
            try:
                if child.find('comment').text == card_type:
                    if int(child.find('team').text) == home_team:
                        stat_home_team += 1
                    else:
                        stat_away_team += 1
            except AttributeError:
                #Some values in the xml doc don't have team values, so there isn't much we can do at this stage
                pass
                
        return stat_home_team, stat_away_team
    
    #Lets take the last possession stat which is available from the xml doc
    if tree.tag == 'possession':
        try:
            last_value = [child for child in tree.iter('value')][-1]
            return int(last_value.find('homepos').text), int(last_value.find('awaypos').text)
        except:
            return None, None
    
    #Taking care of all other stats by extracting based on the home team & away team api id's
    for team in [int(stat.text) for stat in tree.findall('value/team')]:
        if team == home_team: 
            stat_home_team += 1
        else:
            stat_away_team += 1
    return stat_home_team, stat_away_team


# In[48]:


epl.head()


# In[49]:


epl[['on_target_shot_home_team','on_target_shot_away_team']] = epl[['shoton','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['shoton'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")  
epl[['off_target_shot_home_team','off_target_shot_away_team']] = epl[['shotoff','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['shotoff'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand") 
epl[['foul_home_team','foul_away_team']] = epl[['foulcommit','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['foulcommit'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")  
epl[['yellow_card_home_team','yellow_card_away_team']] = epl[['card','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['card'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")
epl[['red_card_home_team','red_card_away_team']] = epl[['card','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['card'],x['home_team_api_id'],x['away_team_api_id'], card_type='r'), axis = 1,result_type="expand")  
epl[['crosses_home_team','crosses_away_team']] = epl[['cross','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['cross'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")  
epl[['corner_home_team','corner_away_team']] = epl[['corner','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['corner'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")  
epl[['possession_home_team','possession_away_team']] = epl[['possession','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['possession'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")


# In[50]:


epl.tail()


# In[ ]:




