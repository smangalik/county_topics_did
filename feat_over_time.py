""" 
run as `python3 feat_over_time.py` 
"""

import os, time, json, datetime, sys

from cycler import cycler

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pandas.plotting import register_matplotlib_converters


from pymysql import cursors, connect
from tqdm import tqdm

register_matplotlib_converters()

# Open default connection
print('Connecting to MySQL...')
connection  = connect(read_default_file="~/.my.cnf")

# Get supplemental data
county_info = pd.read_csv("county_fips_data.csv",encoding = "utf-8")
county_info['fips'] = county_info['fips'].astype(str).str.zfill(5)

# Iterate over all time units to create county_feats[county][year_week] = feats
def get_county_feats(cursor, table_years):
  county_feats = {}
  for table_year in table_years:
    print('Processing {}'.format(table_year))

    sql = "select * from ctlb2.feat$dd_depAnxLex$timelines{}$yw_cnty$1gra;".format(table_year)
    cursor.execute(sql)

    for result in tqdm(cursor.fetchall_unbuffered()): # Read _unbuffered() to save memory
      
      # TODO how do we want to process the value_norm?
      yw_county, feat, value, value_norm = result

      if feat == '_int': continue
      yearweek, county = yw_county.split(":")
      if county == "" or yearweek == "": continue
      county = str(county).zfill(5)

      # Store county_feats
      if county_feats.get(county) is None:
        county_feats[county] = {}
      if county_feats[county].get(yearweek) is None:
        county_feats[county][yearweek] = {}
      county_feats[county][yearweek][feat] = value

  return county_feats

# Returns the first and last day of a week given as "2020_11"
def yearweek_to_dates(yw):
  year, week = yw.split("_")
  year, week = int(year), int(week)

  first = datetime.datetime(year, 1, 1)
  base = 1 if first.isocalendar()[1] == 1 else 8
  monday = first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))
  sunday = monday + datetime.timedelta(days=6)
  return monday, sunday

def county_list_to_df(county_list):
  # Get all available year weeks in order
  available_yws = []
  for county in county_list: 
      available_yws.extend( list(county_feats[county].keys()) )
  available_yws = list(set(available_yws))
  available_yws.sort()
  # print("All available year weeks:",available_yws,"\n")

  # Get feature scores over time
  # yw_anx_score[yearweek] = [ all anx_scores... ]
  yw_anx_score = {}
  yw_dep_score = {}
  
  for county in county_list:
      yearweeks = list(county_feats[county].keys())
      for yearweek in yearweeks:
          # Add anxiety scores              
          if yearweek not in yw_anx_score.keys():
              yw_anx_score[yearweek] = []
          yw_anx_score[yearweek].append( county_feats[county][yearweek]['ANX_SCORE'] )

          # Add depression scores              
          if yearweek not in yw_dep_score.keys():
              yw_dep_score[yearweek] = []
          yw_dep_score[yearweek].append( county_feats[county][yearweek]['DEP_SCORE'] )

  # Store results
  columns = ['date','avg_anx','avg_dep','std_anx','std_dep','n']
  df = pd.DataFrame(columns=columns)

  for yw in available_yws:
      monday, sunday = yearweek_to_dates(yw)

      avg_anx = np.mean(yw_anx_score[yw])
      avg_dep = np.mean(yw_dep_score[yw])
      n = float(min(len(yw_anx_score[yw]), len(yw_dep_score[yw])))
      std_anx = np.std(yw_anx_score[yw])
      std_dep =  np.std(yw_dep_score[yw]) 

      df2 = pd.DataFrame([[monday, avg_anx, avg_dep, std_anx, std_dep, n]], columns=columns)
      df = df.append(df2, ignore_index = True)

  # GROUP BY if necessary
  df.set_index('date', inplace=True) 
  #df = df.groupby(pd.Grouper(freq='Q')).mean() # Q = Quarterly, M = Monthly

  # Calculate columns
  df['ci_anx'] = df['std_anx'] / df['n']**(0.5) # remove sqrt for std-dev
  df['ci_dep'] = df['std_dep'] / df['n']**(0.5) # remove sqrt for std-dev
  df['ci_anx_up'] = df['avg_anx'] + df['ci_anx']
  df['ci_anx_down'] = df['avg_anx'] - df['ci_anx']
  df['ci_dep_up'] = df['avg_dep'] + df['ci_dep']
  df['ci_dep_down'] = df['avg_dep'] - df['ci_dep']

  return df

def plot_depression(counties_of_interest, counties_name, stderr=True):
  counties_of_interest = list(set(county_feats.keys() ) & set(counties_of_interest))
  df = county_list_to_df(counties_of_interest)

  x = df.index.tolist()
  label = 'Average Depression ' + counties_name
  print("Plotting",label)
  plt.plot(x, df['avg_dep'],  label=label)
  if stderr:
    plt.fill_between(x, df['ci_dep_down'].tolist(), df['ci_dep_up'].tolist(), alpha=0.3) # error area

def plot_anxiety(counties_of_interest, counties_name, stderr=True):
  counties_of_interest = list(set(county_feats.keys() ) & set(counties_of_interest))
  df = county_list_to_df(counties_of_interest)

  x = df.index.tolist()
  label = 'Average Anxiety ' + counties_name
  print("Plotting",label)
  plt.plot(x, df['avg_anx'],  label=label)
  if stderr:
    plt.fill_between(x, df['ci_anx_down'].tolist(), df['ci_anx_up'].tolist(), alpha=0.3) # error area

def plot_events():
  plt.axvline(dt.datetime(2019, 12, 25), color="g", label="Christmas") 
  plt.axvline(dt.datetime(2020, 5, 25), color="g", label="George Floyd")
  plt.axvline(dt.datetime(2020, 11, 7), color="g", label="Presidential Election Results") 

with connection:
  with connection.cursor(cursors.SSCursor) as cursor:
    print('Connected to',connection.host)

    # Get county feat information, 
    # county_feats['01087'] = {'2020_34': {'DEP_SCORE': 0.2791998298997296, 'ANX_SCORE': 1.759353260415711}}}
    county_feats_json = "/data/smangalik/county_feats_ctlb.json"
    if not os.path.isfile(county_feats_json):
        #table_years = list(range(2012, 2017))
        table_years = [2019,2020]
        county_feats = get_county_feats(cursor,table_years)
        with open(county_feats_json,"w") as json_file: json.dump(county_feats,json_file)
    start_time = time.time()
    print("\nImporting produced county features\n")
    with open(county_feats_json) as json_file: 
        county_feats = json.load(json_file)

    # Limit county list to one state/region/division at a time
    ny_counties = county_info.loc[county_info['state_abbr'] == "NY", 'fips'].tolist()
    al_counties = county_info.loc[county_info['state_abbr'] == "AL", 'fips'].tolist()
    ca_counties = county_info.loc[county_info['state_abbr'] == "CA", 'fips'].tolist()
    tx_counties = county_info.loc[county_info['state_abbr'] == "TX", 'fips'].tolist()
    oh_counties = county_info.loc[county_info['state_abbr'] == "OH", 'fips'].tolist()
    r1_counties = county_info.loc[county_info['region'] == 1, 'fips'].tolist() # North East
    r2_counties = county_info.loc[county_info['region'] == 2, 'fips'].tolist() # Midwest
    r3_counties = county_info.loc[county_info['region'] == 3, 'fips'].tolist() # South
    r4_counties = county_info.loc[county_info['region'] == 4, 'fips'].tolist() # West
    d1_counties = county_info.loc[county_info['division'] == 1, 'fips'].tolist() # New England
    d9_counties = county_info.loc[county_info['division'] == 9, 'fips'].tolist() # Pacific

    # TODO Top 5 counties by population
    top_pop = ['06037','17031','48201','04013','06073'] # LA, Cook, Harris, Maricopa, San Diego

    regions = [r1_counties,r2_counties,r3_counties,r4_counties]
    region_names = ["in the Northeast","in the Midwest","in the South","in the West"]
    #divisions = [d1_counties,d2_counties,d3_counties,d4_counties,d5_counties,d6_counties,d7_counties,d8_counties,d9_counties]

    all_counties = county_feats.keys() 
    county_list = all_counties
    #county_list = list(set(county_feats.keys() ) & set(ny_counties)) # TODO REMOVE
    #print("Counties considered:", county_list, "\n")
    
    df = county_list_to_df(county_list)    

    #print(df.head())

    # Set up plot
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18, 8)

    x = df.index.tolist()

    # Plot Depression and Anxiety
    anx_line = plt.plot(x, df['avg_anx'], 'b-', label='Average Anxiety')
    dep_line = plt.plot(x, df['avg_dep'], 'r-', label='Average Depression')
    anx_area = plt.fill_between(x, df['ci_anx_down'].tolist(), df['ci_anx_up'].tolist(), color='c', alpha=0.4) # error area
    dep_area = plt.fill_between(x, df['ci_dep_down'].tolist(), df['ci_dep_up'].tolist(), color='pink', alpha=0.4) # error area
    # Events
    plot_events()
    # Make plot pretty
    plt.title("Depression & Anxiety Over Time")
    plt.xlabel("Time")
    plt.ylabel("Feature Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    # Plot everything
    plt.savefig("over_time_depression_and_anxiety.png", bbox_inches='tight')

    # Depression Plot
    plt.clf()
    #plot_depression(ny_counties, "in New York")
    #for region,region_name in zip(regions,region_names): plot_depression(region, region_name, stderr=False)
    plot_depression(d1_counties, "in New England")
    plot_depression(all_counties, "Nationally")
    ax.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) + cycler('lw', [1, 2, 3, 4]))
    # Make plot pretty
    plt.title("Depression Over Time")
    plt.xlabel("Time")
    plt.ylabel("Depression Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    # Plot everything
    plt.savefig("over_time_depression.png", bbox_inches='tight')

    # Anxiety Plot
    plt.clf()
    #plot_anxiety(ny_counties, "in New York")
    #for region,region_name in zip(regions,region_names): plot_anxiety(region, region_name, stderr=False)
    plot_anxiety(d1_counties, "in New England")
    plot_anxiety(all_counties, "Nationally")
    # Make plot pretty
    plt.title("Anxiety Over Time")
    plt.xlabel("Time")
    plt.ylabel("Anxiety Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    # Plot everything
    plt.savefig("over_time_anxiety.png", bbox_inches='tight')

    # Valid Counties Plot
    plt.clf()
    df = county_list_to_df(all_counties)
    x = df.index.tolist()
    plt.plot(x, df['n'], 'g-', label='# of Valid Counties')
    # Make plot pretty
    plt.title("Valid Counties Over Time")
    plt.xlabel("Time")
    plt.ylabel("Valid Counties")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    # Plot everything   
    plt.ylim(ymin=0)
    ax.set_ylim(ymin=0)
    ax.set_ylim(bottom=0)
    plt.draw()
    plt.savefig("over_time_n.png", bbox_inches='tight')

    # Baseline Plot
    plt.clf()
    household_pulse = pd.read_csv("./household_pulse_weekly.csv")
    household_pulse['date'] = household_pulse['yearweek'].apply(lambda yw: yearweek_to_dates(yw)[1])
    print("\nHousehold Pulse\n",household_pulse)

    #household_pulse = household_pulse[household_pulse['date'] < '2021-01-01'] # Trim household_pulse to 2020
    x = household_pulse['date'].tolist()
    anx_line = plt.plot(x, household_pulse['avg(gad2_sum)'], 'b-', label='Worry and Anxiety (0 is best)')
    dep_line = plt.plot(x, household_pulse['avg(phq2_sum)'], 'r-', label='Disinterest and Depression (0 is best)')
    gen_line = plt.plot(x, household_pulse['avg(gen_health)'], 'g-', label='General Health (5 is best)')
    plt.title("Baselines Over Time")
    plt.xlabel("Time")
    plt.ylabel("Feature Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)  
    plt.savefig("over_time_health_baselines.png", bbox_inches='tight')


    

