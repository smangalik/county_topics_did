"""
run as `python3 feat_over_time.py`
"""

import os, time, json, datetime, sys

from cycler import cycler

from pymysql import cursors, connect
from tqdm import tqdm

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels as sm

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Open default connection
print('Connecting to MySQL...')
connection  = connect(read_default_file="~/.my.cnf")

# Get supplemental data
county_info = pd.read_csv("/data/smangalik/county_fips_data.csv",encoding = "utf-8")
county_info['cnty'] = county_info['fips'].astype(str).str.zfill(5)

# Iterate over all time units to create county_feats[county][year_week][DEP_SCORE] = feats
def get_county_feats(cursor, table_years):
  county_feats = {}
  for table_year in table_years:
    print('Processing {}'.format(table_year))

    sql = "select * from ctlb2.feat$dd_depAnxAng$timelines{}$yw_cnty$1gra;".format(table_year)
    cursor.execute(sql)

    for result in tqdm(cursor.fetchall_unbuffered()): # Read _unbuffered() to save memory

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
      county_feats[county][yearweek][feat] = value_norm

  return county_feats

def yearweek_to_dates(yw):
  year, week = yw.split("_")
  year, week = int(year), int(week)

  first = datetime.datetime(year, 1, 1)
  base = 1 if first.isocalendar()[1] == 1 else 8
  monday = first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))
  sunday = monday + datetime.timedelta(days=6)
  thursday = monday + datetime.timedelta(days=3)
  return monday, thursday, sunday

def county_list_to_full_df(county_list):
  rows = []
  for cnty in county_list:
      for yw in list(county_feats[cnty].keys()): # for each valid yw
        year, week = yw.split("_")
        monday, thursday, sunday = yearweek_to_dates(yw)
        yearweek_cnty = "{}:{}".format(yw,cnty)
        year_cnty = "{}:{}".format(year,cnty)
        avg_anx = county_feats[cnty][yw]['ANX_SCORE']
        avg_dep = county_feats[cnty][yw]['DEP_SCORE']

        row = {
          "date":monday,
          'yearweek_cnty':yearweek_cnty,
          'year_cnty':year_cnty,
          'yearweek':yw,
          'year':year,
          'cnty':cnty,
          'avg_anx':avg_anx,
          'avg_dep':avg_dep
        }
        rows.append(row)

  df = pd.DataFrame.from_dict(rows)
  df = pd.merge(df,county_info[['cnty','state_name','region_name']],on='cnty')
  df['yearweek_state'] = df['yearweek'] + ":" + df['state_name']

  # GROUP BY if necessary
  #df.set_index('date', inplace=True)
  #df = df.groupby(pd.Grouper(freq='Q')).mean() # Q = Quarterly, M = Monthly

  return df

with connection:
  with connection.cursor(cursors.SSCursor) as cursor:
    print('Connected to',connection.host)

    # Get county feat information
    county_feats_json = "/data/smangalik/county_feats_ctlb.json"
    if not os.path.isfile(county_feats_json):
        table_years = [2019,2020]
        county_feats = get_county_feats(cursor,table_years)
        with open(county_feats_json,"w") as json_file: json.dump(county_feats,json_file)
    start_time = time.time()
    print("\nImporting produced county features")
    with open(county_feats_json) as json_file:
        county_feats = json.load(json_file)
    print("Import complete\n")
    all_counties = county_feats.keys()
    county_list = all_counties
    print("Counties considered:", len(county_list), "\n")

    # Process Language Based Assessments data
    lba_full = county_list_to_full_df(county_list)
    print("LBA (full)\n",lba_full.head(10),'\nrow count:',len(lba_full));

    # Process Gallup COVID Panel
    sql = "select fips as cnty, yearweek, WEB_worryF, WEC_sadF, neg_affect_lowArousal, neg_affect_highArousal, neg_affect, pos_affect, affect_balance from gallup_covid_panel_micro_poll.old_hasSadBefAug17_recodedEmoRaceGenPartyAge_v3_02_15;"
    gallup_full = pd.read_sql(sql, connection)
    gallup_full = gallup_full[gallup_full['cnty'].isin(all_counties)] # filter to only counties in all_counties
    gallup_full['yearweek'] = gallup_full['yearweek'].astype(str)
    gallup_full['yearweek'] = gallup_full['yearweek'].str[:4] + "_" + gallup_full['yearweek'].str[4:]
    gallup_full['yearweek_cnty'] = gallup_full['yearweek'] + ":" + gallup_full['cnty']
    gallup_full = pd.merge(gallup_full,county_info[['cnty','state_name']],on='cnty')
    gallup_full['yearweek_state'] = gallup_full['yearweek'] + ":" + gallup_full['state_name']
    print("\nGallup COVID Panel (Full)\n",gallup_full.head(10),'\nrow count:',len(gallup_full));

    # Prepare Fixed Effects Data
    gallup_cols = ['WEC_sadF', 'WEB_worryF'] + ['yearweek_cnty']
    data = gallup_full[gallup_cols].merge(lba_full, on='yearweek_cnty')

    # filter data to need at least min_entity_count entries per entity
    entity = "cnty"
    entity_value_counts = data[entity].value_counts()
    min_entity_count = 100
    valid_entities = list(entity_value_counts[entity_value_counts > min_entity_count].index)
    #print("Value Counts:\n",entity_value_counts)
    #print("Valid Entities\n",valid_entities)
    data = data[data[entity].isin(valid_entities)]

    print("\nMerged Data to Calculate Effects\n",data.head(10),'\nrow count:',len(data));

    # Run Mixed Linear Model
    mdf = smf.mixedlm(formula="WEC_sadF ~ avg_dep", data=data, groups=data["cnty"]).fit()
    print('\n',mdf.summary())

    # TODO De-Mean Data
    demean = False
    Y = "WEC_sadF" # WEB_worryF
    T = "avg_dep" # avg_anx
    X = [T,"C(region_name)"] # state_name, region_name, year, yearweek, month, use C(dummy_var) to create dummy vars
    entity = "cnty"

    if demean: # does not support dummy vars
      mean_data = data.groupby(entity)[X+[Y]].mean()
      data = (data
                .set_index(entity) # set the index as the entity indicator
                [X+[Y]]
                - mean_data) # subtract the mean data

    # Run OLS Model
    mod = smf.ols("{} ~ {}".format(Y, '+'.join(X)), data=data).fit()
    print('\t\t\tDe-Meaned OLS\n')
    print(mod.summary().tables[1])
