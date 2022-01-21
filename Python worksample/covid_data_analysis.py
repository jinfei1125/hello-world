'''
This file contain functions for covid data analysis.

Arthor: Jinfei Zhu

Download plotly package:
(To use plotly succesfully, we also need to download nbformat)
pip install nbformat
pip install plotly==4.14.3

Reference:
Regex in pandas: https://kanoki.org/2019/11/12/how-to-use-regex-in-pandas/
Seaborn: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
    https://seaborn.pydata.org/generated/seaborn.relplot.html
plotly: https://plotly.com/python/choropleth-maps/
'''


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
pd.options.mode.chained_assignment = None

def clean_data(df):
    '''
    Clean the raw data:
        1. find the correct state names 
           (some records are not standard at the begining of covid)
        2. Get the correct date (2021/1/1 -> 2021/01/01)
        3. Sort values by state and date
        4. Reset index
        5. Change data type of 'deaths_state'from float to int
    '''
    
    df = df.dropna()
    state_bool = 'province_state' in df.columns
    
    if state_bool:
        state_names =(
            ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", 
            "California", "Colorado", "Connecticut", "District of Columbia", 
            "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", 
            "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", 
            "Louisiana", "Massachusetts","Maryland", "Maine", "Michigan", 
            "Minnesota", "Missouri", "Mississippi", "Montana", 
            "North Carolina", "North Dakota", "Nebraska", "New Hampshire",
             "New Jersey", "New Mexico", "Nevada", "New York", 
            "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", 
            "Rhode Island", "South Carolina", "South Dakota", "Tennessee", 
            "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", 
            "Washington", "Wisconsin", "West Virginia", "Wyoming"])

        # Make sure the state names are legit
        df = df[df.province_state.isin(state_names)]
            
    # Get the correct date
    df = get_correct_date(df)
    
    if state_bool:
        df = df.sort_values(by=['province_state','date'])
    else:
        df = df.sort_values(by=['date'])
        
    df = df.reset_index(drop=True)
    return df

def get_correct_date(df):
    '''
    Change the format of date so we can sort them correctly.
    '''

    replace_pattern = ({r'/1$':'/01',
                r'/2$':'/02',
                r'/3$':'/03',
                r'/4$':'/04',
                r'/5$':'/05',
                r'/6$':'/06',
                r'/7$':'/07',
                r'/8$':'/08',
                r'/9$':'/09',
                r'/1/':'/01/',
                r'/2/':'/02/',
                r'/3/':'/03/',
                r'/4/':'/04/',
                r'/5/':'/05/',
                r'/6/':'/06/',
                r'/7/':'/07/',
                r'/8/':'/08/',
                r'/9/':'/09/'})
    for k, v in replace_pattern.items():
        replaced_series = df['date'].str.replace(k,v, regex=True)
        df['date'] = replaced_series

    return df

def get_the_latest_date(df):
    '''
    Find the latest date in df
    '''
    return sorted(df.date)[-1]

def find_top_n_hardest_hit_state(df, n=5):
    '''
    Find top n hardest by the latest date
    '''

    latest_date = sorted(df.date)[-1]
    latest_confirmed_df = (df[df['date'] == latest_date]
         .sort_values(by='confirmed_state',ascending=False)
         [['province_state','confirmed_state','deaths_state']][:n]
         .reset_index(drop=True))
    return(latest_confirmed_df)

def draw_top_n_hardest_hit_state(df, var='confirmed_state', n=5):
    '''
    Draw the top n hardest hit state by death/confirmed 
    var(str): the variable to be drawn (deaths or confirmed)
    '''
    latest_date = sorted(df.date)[-1]
    top_n_state = list(df[df['date'] == latest_date]
         .sort_values(by=var,ascending=False)
         .province_state[:n])

    # create a data frame contain the time trend of the top n state
    top_state_time_trend_df = df[df.province_state.isin(top_n_state)]

    # clean the name
    if var == 'confirmed_state':
        col_name = 'Confirmed'
    else:
        col_name = 'Deaths'

    # draw the relplot
    sns.relplot(x='date', y=var, kind='line',
                data=top_state_time_trend_df,
                hue='province_state', ci=None)

    plt.xticks(rotation=90)

    if len(top_state_time_trend_df) > 10:
        plt.xticks(ticks=np.arange(0, len(top_state_time_trend_df)/n,
                step=len(top_state_time_trend_df)/n//10),
                 rotation=90)

    plt.xlabel('Date')  
    plt.ylabel('Number')  
    plt.title(label=f"Top {n} States by {col_name} Number") 

    plt.show()



def draw_state_heatmap(df, var='confirmed_state'):
    '''
    Draw a heatmap of the latest date number.
    '''
    df = find_top_n_hardest_hit_state(df, 55)
    df['code']=df['province_state'].apply(lambda x: us_state_abbrev[x])

    if var == 'confirmed_state':
        col_name = 'Confirmed'
        color = 'Reds'
    else:
        col_name = 'Deaths'
        color = 'Greys'

    fig = go.Figure(data=go.Choropleth(
    locations=df['code'], # Spatial coordinates
    z = df[var].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = color,
    colorbar_title = f"{col_name} Number",
    ))

    fig.update_layout(
        title_text = f'COVID-19 {col_name} Number by State',
        geo_scope='usa', # limite map scope to USA
        )

    fig.show()


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}



