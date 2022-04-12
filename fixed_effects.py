"""
run as `python3 fixed_effects.py`
"""

import matplotlib
matplotlib.use('Agg')

import os, time, json, datetime, sys

from pymysql import cursors, connect
from tqdm import tqdm

import seaborn as sns

import pandas as pd
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
    #county_feats_json = "/data/smangalik/county_feats_ctlb_30user.json" # /data/smangalik/county_feats_ctlb_X0user.json
    
    #county_feats_json = "/data/smangalik/county_feats_ctlb_std.json" # standardized experiment
    county_feats_json = "/data/smangalik/county_feats_ctlb_nostd.json" # non-standardized experiment
    #county_feats_json = "/data/smangalik/county_feats_ctlb_nofs.json" # no feature selection experiment
    
    print("Running on",county_feats_json)
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
    gallup = pd.read_sql(sql, connection)
    gallup = gallup[gallup['cnty'].isin(all_counties)] # filter to only counties in all_counties
    gallup['yearweek'] = gallup['yearweek'].astype(str)
    gallup['yearweek'] = gallup['yearweek'].str[:4] + "_" + gallup['yearweek'].str[4:]
    gallup['yearweek_cnty'] = gallup['yearweek'] + ":" + gallup['cnty']
    gallup = gallup.groupby("yearweek_cnty").mean().reset_index() # mean aggregate through yearweek_county
    gallup[['yearweek','cnty']] = gallup['yearweek_cnty'].str.split(":",expand=True)
    gallup = pd.merge(gallup,county_info[['cnty','state_name','region_name']],on='cnty')
    gallup['yearweek_state'] = gallup['yearweek'] + ":" + gallup['state_name']
    print("\nGallup COVID Panel (AVG on yearweek_cnty)\n",gallup.head(10),'\nrow count:',len(gallup));

    # Prepare Fixed Effects Data
    gallup_cols = ['WEC_sadF', 'WEB_worryF'] + ['yearweek_cnty']
    data = gallup[gallup_cols].merge(lba_full, on='yearweek_cnty')

    # filter data to need at least min_entity_count entries per entity
    entity = "cnty"
    entity_value_counts = data[entity].value_counts()
    min_entity_count = 5
    valid_entities = list(entity_value_counts[entity_value_counts > min_entity_count].index)
    #print("Value Counts:\n",entity_value_counts)
    #print("Valid Entities\n",valid_entities)
    data = data[data[entity].isin(valid_entities)]
    print("\nMerged LBA and Gallup\n",data.head(10),'\nrow count:',len(data));

    # De-Mean Data
    Y = "WEC_sadF" # WEB_worryF
    T = "avg_dep" # avg_anx
    X = [T,"region_name"] # state_name, region_name, year, yearweek, month, use C(dummy_var) to create dummy vars
    entity = "cnty"

    # De-Mean All The Outcomes
    outcomes = ['avg_dep','avg_anx','WEC_sadF','WEB_worryF']
    avg_outcomes = ["avg({})".format(x) for x in outcomes]
    mean_data = data.groupby(entity)[outcomes].mean().reset_index() # find average within each entity
    mean_data.columns = ["cnty"]+avg_outcomes
    demeaned_data = data.merge(mean_data, on='cnty') # merge on the average values
    demeaned_data[outcomes] = demeaned_data[outcomes] - demeaned_data[avg_outcomes].values # subtract off the average values
    demeaned_data = demeaned_data.drop(columns=avg_outcomes)

    # TODO confirm if standardizing the demeaned data is good
    cols_to_std = []
    #cols_to_std = ['WEB_worryF','WEC_sadF','avg_anx','avg_dep']
    for col in cols_to_std:
      demeaned_data[col] = demeaned_data[col] / demeaned_data[col].std()

    print("\nDemeaned Data\n",demeaned_data.head(10),'\nrow count:',len(demeaned_data));

    print("\nDemeaned Data Stats")
    print("demeaned_data['WEB_worryF'].std()",demeaned_data['WEB_worryF'].std())
    print("demeaned_data['WEC_sadF'].std()",demeaned_data['WEC_sadF'].std())
    print("demeaned_data['avg_anx'].std()",demeaned_data['avg_anx'].std())
    print("demeaned_data['avg_dep'].std()",demeaned_data['avg_dep'].std())
    print()

    # Run OLS Model
    #formula = "WEC_sadF ~ avg_dep + region_name" # change this to alter fixed effects equation
    formula = "WEB_worryF ~ avg_anx + region_name" # change this to alter fixed effects equation
    #formula = "WEC_sadF ~ avg_anx + region_name" # change this to alter fixed effects equation
    #formula = "WEB_worryF ~ avg_dep + region_name" # change this to alter fixed effects equation
    print('\t\t\tDe-Meaned OLS on',formula)
    mod = smf.ols(formula, data=demeaned_data).fit()
    print(mod.summary().tables[1])

    # Run correlations (can be done on any level)
    group_on = 'yearweek_cnty' # yearweek = week x national, yearweek_cnty = week x county, yearweek_state = week x state, cnty = year x county
    merge = demeaned_data.groupby(group_on).mean().reset_index()
    print("\nAggregated Data for Correlation\n",merge.head(5))
    corr = merge.corr(method="pearson")
    print("\nLBA vs Gallup COVID by",group_on,'\n', corr)
    print(len(merge),"samples used for correlation")
    corr_plot = sns.heatmap(corr.head(2), center=0, square=True, linewidths=.5, annot=True)
    corr_plot.figure.savefig("LBA vs Gallup Demeaned.png", bbox_inches='tight')

    # Run Mixed Linear Model
    # md = smf.mixedlm(formula="WEC_sadF ~ avg_dep + region_name", data=data, groups=data["cnty"])
    # mdf = md.fit()
    # print('\n',mdf.summary())

    # Make Gallup into feat table: ['group_id', 'feat', 'value', 'group_norm']
    gallup_worry = gallup.copy(deep=True)
    gallup_worry['feat'] = "WEB_worryF"
    gallup_worry['value'] = gallup_worry['WEB_worryF']
    gallup_sad = gallup.copy(deep=True)
    gallup_sad['feat'] = "WEC_sadF"
    gallup_sad['value'] = gallup_worry['WEC_sadF']
    gallup_mysql = pd.concat([gallup_worry, gallup_sad])
    gallup_mysql = gallup_mysql.rename(columns={"yearweek_cnty":"group_id"})
    gallup_mysql["group_norm"] = gallup_mysql["value"]
    gallup_mysql = gallup_mysql[['group_id', 'feat', 'value', 'group_norm']]
    print(gallup_mysql)

    from sqlalchemy import create_engine
    from sqlalchemy.engine.url import URL
    myDB = URL(drivername='mysql', host='localhost',
      database='ctlb2', query={ 'read_default_file' : "~/.my.cnf" }
    )
    engine = create_engine(name_or_url=myDB)

    gallup_mysql.to_sql("feat$gallup_covid$yw_cnty",con=engine, if_exists='replace', index=False)