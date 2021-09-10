""" 
run as `python3 feat_over_time.py` 
"""

import os, time, json, datetime

import numpy as np
import matplotlib.pyplot as plt

from pymysql import cursors, connect
from tqdm import tqdm


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

    # Plot results
    x = []
    avg_anxs = []
    avg_deps = []
    ci_anx_ups = []
    ci_anx_downs = []
    ci_dep_ups = []
    ci_dep_downs = []
    for yw in available_yws:
        monday, sunday = yearweek_to_dates(yw)
        avg_anx = np.mean(yw_anx_score[yw])
        avg_dep = np.mean(yw_dep_score[yw])
        ci_anx = np.std(yw_anx_score[yw]) # / len(yw_anx_score[yw]) 
        ci_dep =  np.std(yw_dep_score[yw]) # / len(yw_dep_score[yw]) 

        x.append(sunday)
        avg_anxs.append(avg_anx)
        avg_deps.append(avg_dep)
        ci_anx_ups.append(avg_anx + ci_anx)
        ci_anx_downs.append(avg_anx - ci_anx)
        ci_dep_ups.append(avg_dep + ci_dep)
        ci_dep_downs.append(avg_dep - ci_dep)

    # plot results
    plt.plot(x, avg_anxs, 'b-', label='Average Anxiety')
    #plt.plot(x, avg_deps, 'r-', label='Average Depression')
    plt.fill_between(x, ci_anx_downs, ci_anx_ups, color='c', alpha=0.2)
    #plt.fill_between(x, ci_dep_downs, ci_dep_ups, color='pink', alpha=0.2)

    # Make plot pretty
    plt.title("Depression/Anxiety Over Time")
    plt.xlabel("Time")
    plt.ylabel("Feature Score")
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()
    plt.show()


    

