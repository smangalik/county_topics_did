"""
run as `python3 feat_over_time.py`
"""

import os, time, json, datetime, sys

from cycler import cycler

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
  thursday = monday + datetime.timedelta(days=3)
  return monday, thursday, sunday

def county_list_to_df(county_list):
  # Get all available year weeks in order
  available_yws = []
  for county in county_list:
      available_yws.extend( list(county_feats[county].keys()) )
  available_yws = list(set(available_yws))
  available_yws.sort()

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
  columns = ['date','yearweek','avg_anx','avg_dep','std_anx','std_dep','n']
  df = pd.DataFrame(columns=columns)

  for yw in available_yws:
      monday, thursday, sunday = yearweek_to_dates(yw)

      avg_anx = np.mean(yw_anx_score[yw])
      avg_dep = np.mean(yw_dep_score[yw])
      n = float(min(len(yw_anx_score[yw]), len(yw_dep_score[yw])))
      std_anx = np.std(yw_anx_score[yw])
      std_dep =  np.std(yw_dep_score[yw])

      df2 = pd.DataFrame([[monday, yw, avg_anx, avg_dep, std_anx, std_dep, n]], columns=columns)
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
  ax.plot(x, df['avg_dep'],  label=label)
  if stderr:
    ax.fill_between(x, df['ci_dep_down'].tolist(), df['ci_dep_up'].tolist(), alpha=0.3) # error area

def plot_anxiety(counties_of_interest, counties_name, stderr=True):
  counties_of_interest = list(set(county_feats.keys() ) & set(counties_of_interest))
  df = county_list_to_df(counties_of_interest)

  x = df.index.tolist()
  label = 'Average Anxiety ' + counties_name
  print("Plotting",label)
  ax.plot(x, df['avg_anx'],  label=label)
  if stderr:
    ax.fill_between(x, df['ci_anx_down'].tolist(), df['ci_anx_up'].tolist(), alpha=0.3) # error area

def plot_events():
  # plt.axvline(dt.datetime(2019, 12, 25), color="g", label="Christmas") # vertical line
  ax.axvspan(dt.datetime(2019, 12, 25), dt.datetime(2019, 12, 29), alpha=0.3, color='g', label="Christmas 2019")
  ax.axvspan(dt.datetime(2020, 1, 21), dt.datetime(2020, 1, 28), alpha=0.3, color='g', label="First US Case COVID")
  ax.axvspan(dt.datetime(2020, 5, 25), dt.datetime(2020, 5, 30), alpha=0.3, color='g', label="George Floyd")
  ax.axvspan(dt.datetime(2020, 11, 7), dt.datetime(2020, 11, 14), alpha=0.3, color='g', label="Presidential Election Results")

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
    print("\nImporting produced county features")
    with open(county_feats_json) as json_file:
        county_feats = json.load(json_file)
    print("Import complete\n")

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

    print(df.head(10),'\n')

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
    plt.title("National Depression & Anxiety Over Time")
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
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18, 8)
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
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18, 8)
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
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18, 8)

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

    # Household Pulse Plot
    plt.clf()
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18, 8)
    ax2 = ax.twinx()
    household_pulse = pd.read_csv("./household_pulse_weekly.csv")
    household_pulse['date'] = household_pulse['yearweek'].apply(lambda yw: yearweek_to_dates(yw)[1])
    print("\nHousehold Pulse\n",household_pulse.head(10))

    household_pulse = household_pulse[household_pulse['date'] < '2021-01-01'] # Trim household_pulse to 2020
    x = household_pulse['date'].tolist()
    anx_line = ax2.plot(x, household_pulse['avg(gad2_sum)'], 'b-', label='Worry and Anxiety (0 is best)')
    dep_line = ax2.plot(x, household_pulse['avg(phq2_sum)'], 'r-', label='Disinterest and Depression (0 is best)')
    gen_line = ax2.plot(x, household_pulse['avg(gen_health)'], 'g-', label='General Health (5 is best)')
    plot_events()
    plot_depression(all_counties, "Nationally")
    plot_anxiety(all_counties, "Nationally")
    plt.title("Baselines Over Time")
    plt.xlabel("Time")
    plt.ylabel("Feature Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    plt.savefig("over_time_health_baselines.png", bbox_inches='tight')


    # NOT A LOT OF DATA: Gallup Micro-Poll Plot
    plt.clf()
    gallup_old = pd.read_csv("./gallup_micro_polls_weekly.csv")
    gallup_old['date'] = gallup_old['yearweek'].apply(lambda yw: yearweek_to_dates(yw)[1])
    print("\nGallup Micropoll\n",gallup_old.head(10))

    x = gallup_old['date'].tolist()
    pain_line = plt.plot(x, gallup_old['avg(wp68_clean)'], label='Experienced Pain? (0 is best)')
    worry_line = plt.plot(x, gallup_old['avg(wp69_clean)'], label='Experienced Worry? (0 is best)')
    stress_line = plt.plot(x, gallup_old['avg(wp71_clean)'], label='Experienced Stress? (0 is best)')
    dep_line = plt.plot(x, gallup_old['avg(H4D_clean)'], label='Depression Diagnosis? (0 is best)')
    plt.title("Baselines Over Time")
    plt.xlabel("Time")
    plt.ylabel("Feature Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    #dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    dates= list(pd.date_range('2018-11-01','2019-08-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    plt.savefig("over_time_gallup_old.png", bbox_inches='tight')

    # Gallup COVID Panel Plot
    plt.clf()
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18, 8)
    ax2 = ax.twinx()
    sql = "select fips, yearweek, wp16, wp18, WEA_enjoyF, WEB_worryF, WEC_sadF, WED_stressF, WEE_angerF, WEF_happinessF, WEG_boredomF, WEH_lonelyF, WEI_depressionF, WEJ_anxietyF from gallup_covid_panel_micro_poll.old_hasSadBefAug17_recodedEmoRaceGenPartyAge_v3_02_15;"
    gallup = pd.read_sql(sql, connection)
    # TODO do the group by in the sql
    gallup['yearweek'] = gallup['yearweek'].astype(str)
    gallup['yearweek'] = gallup['yearweek'].str[:4] + "_" + gallup['yearweek'].str[4:]
    gallup = gallup.groupby(by=["yearweek"]).mean().reset_index()
    gallup['date'] = gallup['yearweek'].apply(lambda yw: yearweek_to_dates(yw)[1])
    print("\nGallup COVID Panel\n",gallup.head(10))

    x = gallup['date'].tolist()
    sad_line = ax2.plot(x, gallup['WEC_sadF'], label='Sadness')
    worry_line = ax2.plot(x, gallup['WEB_worryF'], label='Worry')
    anx_line = ax2.plot(x, gallup['WEJ_anxietyF'], label='Anxiety')
    dep_line = ax2.plot(x, gallup['WEI_depressionF'], label='Depression')
    plot_events()
    plot_depression(all_counties, "Nationally")
    plot_anxiety(all_counties, "Nationally")
    plt.title("Baselines Over Time")
    plt.xlabel("Time")
    plt.ylabel("Feature Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    #dates= list(pd.date_range('2020-03-01','2020-09-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    plt.savefig("over_time_gallup_covid.png", bbox_inches='tight')


    # CDC BRFSS Plot
    plt.clf()
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18, 8)
    ax2 = ax.twinx()
    brfss_files = ["/data/smangalik/BRFSS_mental_health_2019.csv","/data/smangalik/BRFSS_mental_health_2020.csv"]
    brfss = pd.concat((pd.read_csv(f) for f in brfss_files))
    print("\nCDC BRFSS\n",brfss.head(8))
    brfss = brfss.rename(columns={"YEARWEEK": "yearweek"})
    brfss['DATE'] = pd.to_datetime(brfss['DATE'], infer_datetime_format=True) # infer datetime
    # print('\nyearweek counts\n',brfss['yearweek'].value_counts()) # yearweek data point counts
    brfss = brfss.groupby(by=["yearweek"]).mean().reset_index()
    brfss['DATE'] = brfss['yearweek'].apply(lambda yw: yearweek_to_dates(yw)[1]) # replace date based on yearweek
    print(brfss.head(8))

    x = brfss['DATE'].tolist()
    menthlth_line = ax2.plot(x, brfss['MENTHLTH'], label='Mentally Unhealthy Days (0 is best)',color='r')
    poorhlth_line = ax2.plot(x, brfss['POORHLTH'], label='Health Affected Activities (0 is best)',color='g')
    plot_events()
    plot_depression(all_counties, "Nationally")
    plot_anxiety(all_counties, "Nationally")
    plt.title("Baselines Over Time")
    plt.xlabel("Time")
    plt.ylabel("Days")
    plt.gcf().autofmt_xdate()
    plt.legend()
    dates= list(pd.date_range('2019-01-01','2021-01-01' , freq='1M')-pd.offsets.MonthBegin(1))
    plt.xticks(dates)
    plt.savefig("over_time_brfss.png", bbox_inches='tight')

    # Show correlations
    lba_cols = ['yearweek','avg_anx','avg_dep']
    household_cols = ['yearweek','avg(gen_health)', 'avg(gad2_sum)', 'avg(phq2_sum)']
    gallup_cols = ['yearweek','WEC_sadF', 'WEB_worryF', 'WEJ_anxietyF', 'WEI_depressionF']
    brfss_cols = ['yearweek','MENTHLTH',  'POORHLTH',  'ACEDEPRS', '_MENT14D']
    method="spearman"

    plt.clf()
    corr = df[lba_cols].merge(household_pulse[household_cols], on='yearweek').corr(method=method)
    print("\nLBA vs Household Pulse\n", corr)
    corr_plot = sns.heatmap(corr, center=0, square=True, linewidths=.5, annot=True)
    corr_plot.figure.savefig("LBA vs Household Pulse.png", bbox_inches='tight')

    plt.clf()
    corr = df[lba_cols].merge(gallup[gallup_cols], on='yearweek').corr(method=method)
    print("\nLBA vs Gallup\n", corr)
    corr_plot = sns.heatmap(corr, center=0, square=True, linewidths=.5, annot=True)
    corr_plot.figure.savefig("LBA vs Gallup.png", bbox_inches='tight')

    plt.clf()
    corr = df[lba_cols].merge(brfss[brfss_cols], on='yearweek').corr(method=method)
    print("\nLBA vs BRFSS\n", corr)
    corr_plot = sns.heatmap(corr, center=0, square=True, linewidths=.5, annot=True)
    corr_plot.figure.savefig("LBA vs BRFSS.png", bbox_inches='tight')
