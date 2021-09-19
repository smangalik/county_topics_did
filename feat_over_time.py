""" 
run as `python3 feat_over_time.py` 
"""

import os, time, json, datetime, sys

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

# Iterate over all time units to create county_feats[county][year_week] = feats
def get_county_feats(cursor, table_years):
  county_feats = {}
  for table_year in table_years:
    print('Processing {}'.format(table_year))

    sql = "select * from ctlb2.feat$dd_depAnxLex$timelines{}$yw_cnty$1gra;".format(table_year)
    cursor.execute(sql)

    for result in tqdm(cursor.fetchall_unbuffered()): # Read _unbuffered() to save memory
      
      # TODO do we want to use the values directly or the value norms?
      # TODO if we use value directly do we want to clip it?
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

    # TODO do we need to scale the features by population or is it better to not?

    county_list = county_feats.keys() 

    # Get all available year weeks in order
    available_yws = []
    for county in county_list:
        available_yws.extend( list(county_feats[county].keys()) )
    available_yws = list(set(available_yws))
    available_yws.sort()
    print("All available year weeks:",available_yws,"\n")

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

        # x.append(monday)
        # avg_anxs.append(avg_anx)
        # avg_deps.append(avg_dep)
        # ci_anx_ups.append(avg_anx + ci_anx)
        # ci_anx_downs.append(avg_anx - ci_anx)
        # ci_dep_ups.append(avg_dep + ci_dep)
        # ci_dep_downs.append(avg_dep - ci_dep)

        df2 = pd.DataFrame([[monday, avg_anx, avg_dep, std_anx, std_dep, n]], columns=columns)
        df = df.append(df2, ignore_index = True)

    # GROUP BY if necessary
    df.set_index('date', inplace=True) 
    #df = df.groupby(pd.Grouper(freq='Q')).mean()

    # Calculate columns
    df['ci_anx'] = df['std_anx'] / df['n']**(0.5)
    df['ci_dep'] = df['std_dep'] / df['n']**(0.5)
    df['ci_anx_up'] = df['avg_anx'] + df['ci_anx']
    df['ci_anx_down'] = df['avg_anx'] - df['ci_anx']
    df['ci_dep_up'] = df['avg_dep'] + df['ci_dep']
    df['ci_dep_down'] = df['avg_dep'] - df['ci_dep']

    print(df.head())

    # Set up plot
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18, 8)

    # Create plots
    x = df.index.tolist()
    anx_line = plt.plot(x, df['avg_anx'], 'b-', label='Average Anxiety')
    dep_line = plt.plot(x, df['avg_dep'], 'r-', label='Average Depression')
    anx_area = plt.fill_between(x, df['ci_anx_down'].tolist(), df['ci_anx_up'].tolist(), color='c', alpha=0.4) # error area
    dep_area = plt.fill_between(x, df['ci_dep_down'].tolist(), df['ci_dep_up'].tolist(), color='pink', alpha=0.4) # error area

    # Make plot pretty
    plt.title("Depression/Anxiety Over Time")
    plt.xlabel("Time")
    plt.ylabel("Feature Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)

    # plot everything
    plt.savefig("over_time_depression_and_anxiety.png", bbox_inches='tight')

    # remove anxiety
    trash = anx_line.pop(0)
    trash.remove()
    ax.collections.remove(anx_area)
    ax.relim()
    ax.autoscale()
    plt.draw()
    plt.savefig("over_time_depression.png", bbox_inches='tight')

    # remove depression, add anxiety 
    trash = dep_line.pop(0)
    trash.remove()
    ax.collections.remove(dep_area)
    anx_line = plt.plot(x, df['avg_anx'], 'b-', label='Average Anxiety')
    anx_area = plt.fill_between(x, df['ci_anx_down'].tolist(), df['ci_anx_up'].tolist(), color='c', alpha=0.4)
    ax.relim()
    ax.autoscale()
    plt.draw()
    plt.savefig("over_time_anxiety.png", bbox_inches='tight')

    # remove anxiety, add n 
    trash = anx_line.pop(0)
    trash.remove()
    ax.collections.remove(anx_area)
    plt.plot(x, df['n'], 'g-', label='n')
    ax.relim()
    ax.autoscale()
    plt.draw()
    plt.savefig("over_time_n.png", bbox_inches='tight')

    # Baseline plot
    plt.clf()
    household_pulse = pd.read_csv("./household_pulse_weekly.csv")
    household_pulse['yearweek'] = "2020_" + household_pulse['week'].astype(str).str.zfill(2)
    household_pulse['date'] = household_pulse['yearweek'].apply(lambda yw: yearweek_to_dates(yw)[1])
    print("\n",household_pulse)
    x = household_pulse['date'].tolist()
    anx_line = plt.plot(x, household_pulse['avg(gad2_sum)'], 'b-', label='Worry and Anxiety (0 is best)')
    dep_line = plt.plot(x, household_pulse['avg(phq2_sum)'], 'r-', label='Disinterest and Depression (0 is best)')
    dep_line = plt.plot(x, household_pulse['avg(gen_health)'], 'g-', label='General Health (5 is best)')
    plt.title("Baselines Over Time")
    plt.xlabel("Time")
    plt.ylabel("Feature Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    plt.savefig("health_baselines.png", bbox_inches='tight')


    

