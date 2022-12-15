"""
run as `python3 fixed_effects.py`
"""

import matplotlib
matplotlib.use('Agg')

import os, time, json, datetime, sys

from pymysql import cursors, connect
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels as sm

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

GFT = 200 # now called CUT

# Open default connection
print('Connecting to MySQL...')
connection  = connect(read_default_file="~/.my.cnf")

# Get supplemental data
county_info = pd.read_csv("/data/smangalik/county_fips_data.csv",encoding = "utf-8")
county_info['cnty'] = county_info['fips'].astype(str).str.zfill(5)

counties_we_care_about = ['01073','48113','36103']

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
        avg_anx = county_feats[cnty][yw]['ANX_SCORE']['score']
        avg_dep = county_feats[cnty][yw]['DEP_SCORE']['score']
        std_anx = county_feats[cnty][yw]['ANX_SCORE']['std_score']
        std_dep = county_feats[cnty][yw]['DEP_SCORE']['std_score']

        row = {
          "date":thursday,
          'yearweek_cnty':yearweek_cnty,
          'year_cnty':year_cnty,
          'yearweek':yw,
          'year':year,
          'cnty':cnty,
          'avg_anx':avg_anx,
          'avg_dep':avg_dep,
          'std_anx':std_anx,
          'std_dep':std_dep
        }
        rows.append(row)

  df = pd.DataFrame.from_dict(rows)
  df = pd.merge(df,county_info[['cnty','state_name','region_name']],on='cnty')
  df['yearweek_state'] = df['yearweek'] + ":" + df['state_name']

  # GROUP BY if necessary
  #df.set_index('date', inplace=True)
  #df = df.groupby(pd.Grouper(freq='Q')).mean() # Q = Quarterly, M = Monthly

  return df  

def df_printout(df, outcomes=None, space=None, time=None):
  print(df.head())
  print('N =',len(df),end="")
  if space:
    print(" over {} {}s".format(df[space].nunique(),space),end="")
  if time:
    print(" over {} {}s".format(df[time].nunique(),time),end="")
  print()
  if outcomes:
    for outcome in outcomes:
        print("-> avg({}) \t= {} ({})".format(outcome, df[outcome].mean(), df[outcome].std()))
  print()


with connection:
  with connection.cursor(cursors.SSCursor) as cursor:
    print('Connected to',connection.host)

    #county_feats_json = "/data/smangalik/feat_dd_depAnxLex_19to20_3upt{}user05fc_ywcnty.json".format(GFT) #16to16
    county_feats_json = "/data/smangalik/feat_dd_daa_c2adpt_ans_19to20_3upt{}user05fc_ywcnty.json".format(GFT) #16to8
    
    
    print("Running on",county_feats_json)
    if not os.path.isfile(county_feats_json):
      print("County data not available")
      sys.exit(1)
    print("\nImporting produced county features")
    with open(county_feats_json) as json_file:
        county_feats = json.load(json_file)
    all_counties = county_feats.keys()
    county_list = all_counties

    # Process Language Based Assessments data
    lba_full = county_list_to_full_df(county_list)

    #  [Optional] 2020 minus 2019
    lba_full_2019 = lba_full[lba_full['yearweek'].str.startswith("2019")] # Separate Years
    lba_full_2020 = lba_full[lba_full['yearweek'].str.startswith("2020")]
    lba_full_2019['week'] = lba_full_2019['yearweek'].str.split(":",expand=True)[0].str.split("_",expand=True)[1] # Add week column
    lba_full_2020['week'] = lba_full_2020['yearweek'].str.split(":",expand=True)[0].str.split("_",expand=True)[1]
    lba_full_diff = lba_full_2020.merge(lba_full_2019[['cnty','week','avg_anx','avg_dep']], on=['cnty','week'], suffixes= ("_2020", "_2019")) # Merge 2019 yearweek scores onto 2020
    lba_full_diff['avg_anx'] =  lba_full_diff['avg_anx_2020'] - lba_full_diff['avg_anx_2019'] # scoreCol = scoreCol_2020 - scoreCol_2019
    lba_full_diff['avg_dep'] =  lba_full_diff['avg_dep_2020'] - lba_full_diff['avg_dep_2019']
    lba_full = lba_full_diff # TODO change this to lba_full_diff to get 2020-2019

    print("LBA (full)\n",lba_full.head(10))
    print('N =',len(lba_full),'covering',lba_full['cnty'].nunique(),'counties and',lba_full['yearweek'].nunique(),'weeks\n');
    #print("min =",lba_full['avg_dep'].min(),"; max =",lba_full['avg_dep'].max(), "std =",lba_full['avg_dep'].std())

    # Process Gallup COVID Panel
    gft = 0
    gallup_gft = gft * 22 # GFT * 22 weeks
    sql = "select fips as cnty, yearweek, WEB_worryF, WEC_sadF, neg_affect_lowArousal, neg_affect_highArousal, neg_affect, pos_affect, affect_balance from gallup_covid_panel_micro_poll.old_hasSadBefAug17_recodedEmoRaceGenPartyAge_v3_02_15;"
    gallup = pd.read_sql(sql, connection)
    print('Gallup by yw_user: N =',\
      len(gallup),'covering',gallup['cnty'].nunique(),'counties and',gallup['yearweek'].nunique(),'weeks');
    gallup_county_counts = gallup['cnty'].value_counts()
    gallup_passing_counties = list(gallup_county_counts[gallup_county_counts >= gallup_gft].index)
    gallup = gallup[gallup['cnty'].isin(gallup_passing_counties)]
    gallup = gallup.groupby(by=["yearweek","cnty"]).mean().reset_index() # aggregate to yearweek_cnty so we share group_id with LBA
    gallup['yearweek'] = gallup['yearweek'].astype(str)
    gallup['yearweek'] = gallup['yearweek'].str[:4] + "_" + gallup['yearweek'].str[4:]
    gallup['yearweek_cnty'] = gallup['yearweek'] + ":" + gallup['cnty']
    gallup = pd.merge(gallup,county_info[['cnty','state_name','region_name']],on='cnty')
    gallup['yearweek_state'] = gallup['yearweek'] + ":" + gallup['state_name']
    print("\nGallup COVID Panel (AVG by yearweek_cnty)\n",gallup.head(10))
    print("Gallup by yw_cnty w/ {}GFT: N =".format(gallup_gft),\
      len(gallup),'covering',gallup['cnty'].nunique(),'counties and',gallup['yearweek'].nunique(),'weeks');

    # Prepare Fixed Effects Data
    gallup_cols = ['WEC_sadF', 'WEB_worryF'] + ['yearweek_cnty']
    data = gallup[gallup_cols].merge(lba_full, on='yearweek_cnty')
    print("\nMerged LBA and Gallup\n",data.head())
    print('N =',len(data),'covering',data['cnty'].nunique(),'counties and',data['yearweek'].nunique(),'weeks');

    # Filter to counties with full coverage of weeks
    counties_full_coverage = data['cnty'].value_counts()[data['cnty'].value_counts() == data['yearweek'].nunique()].index.to_list()
    data = data[data['cnty'].isin(counties_full_coverage)]
    print("\nEnforce Full Week Coverage\n",data.head())
    print('N =',len(data),'covering',data['cnty'].nunique(),'counties and',data['yearweek'].nunique(),'weeks');

    outcomes = ['avg_dep','avg_anx','WEC_sadF','WEB_worryF']
    print("\nData Stats: mean (std)")
    for outcome in outcomes:
        print("-> {} \t= {} ({})".format(outcome, data[outcome].mean(), data[outcome].std()))

    # County Center All The Outcomes
    entity = "cnty"
    avg_outcomes = ["avg({})".format(x) for x in outcomes]
    mean_data = data.groupby(entity)[outcomes].mean().reset_index() # find average within each entity
    # Apply mean centering to data
    print("\nMean outcome data per {}\n{}".format(entity, mean_data.head()))
    mean_data.columns = [entity]+avg_outcomes
    county_centered_data = data.merge(mean_data, on=entity) # merge on the average values
    county_centered_data[outcomes] = county_centered_data[outcomes] - county_centered_data[avg_outcomes].values # subtract off the average values
    county_centered_data = county_centered_data.drop(columns=avg_outcomes)

    # standardizing the county centered cols
    # cols_to_std = []
    # cols_to_std = ['WEB_worryF','WEC_sadF','avg_anx','avg_dep']
    # for col in cols_to_std:
    #   county_centered_data[col] = county_centered_data[col] / county_centered_data[col].std()
    #   print("county_centered_data['{}'].std() =".format(col), county_centered_data[col].std())
    # # Scale results to [0,1]
    # min_max_scaler = MinMaxScaler()
    # county_centered_data[cols_to_std] = min_max_scaler.fit_transform(county_centered_data[cols_to_std])
    # data[cols_to_std] = min_max_scaler.fit_transform(data[cols_to_std])

    # Add month and quarter to mean centered data
    county_centered_data['month'] = pd.DatetimeIndex(county_centered_data['date']).month
    county_centered_data['quarter'] = pd.DatetimeIndex(county_centered_data['date']).quarter

    print("\nCounty Centered Data")
    df_printout(county_centered_data,outcomes,space="cnty",time="yearweek")

    # Aggregate to County-Month
    county_centered_data_countymonth = county_centered_data.groupby(["cnty","month"]).mean().reset_index()
    print("County Month County Centered Data")
    df_printout(county_centered_data_countymonth,outcomes,space="cnty",time="month")

    # Aggregate to County-Quarter
    county_centered_data_countyquarter = county_centered_data.groupby(["cnty","quarter"]).mean().reset_index()
    print("County Quarter County Centered Data")
    df_printout(county_centered_data_countyquarter,outcomes,space="cnty",time="quarter")

    # Aggregate to Region-Week
    county_centered_data_regionweek = county_centered_data.groupby(["region_name","yearweek"]).mean().reset_index()
    print("Region Week County Centered Data")
    df_printout(county_centered_data_regionweek,outcomes,space="region_name",time="yearweek")

    # Aggregate to Region-Month
    county_centered_data_regionmonth = county_centered_data.groupby(["region_name","month"]).mean().reset_index()
    print("Region Month County Centered Data")
    df_printout(county_centered_data_regionmonth,outcomes,space="region_name",time="month")

     # Aggregate to Region-Quarter
    county_centered_data_regionquarter = county_centered_data.groupby(["region_name","quarter"]).mean().reset_index()
    print("Region Quarter County Centered Data")
    df_printout(county_centered_data_regionquarter,outcomes,space="region_name",time="quarter")

    # Aggregate to National-Week
    county_centered_data_nationalweek = county_centered_data.groupby(["yearweek"]).mean().reset_index()
    print("National Week County Centered Data")
    df_printout(county_centered_data_nationalweek,outcomes,space=None,time="yearweek")

    # Aggregate to County-Year
    county_centered_data_countyyear = data.groupby(["cnty"]).mean().reset_index()
    print("County Year County Centered Data")
    df_printout(county_centered_data_countyyear,outcomes,space="cnty",time=None)
    
    # TODO Using an aggregate?
    county_centered_data = county_centered_data_regionquarter

    # Run OLS Model
    formulas = []
    formulas.append("WEC_sadF ~ avg_dep ") 
    formulas.append("WEB_worryF ~ avg_anx ") 
    formulas.append("WEC_sadF ~ avg_anx " )
    formulas.append("WEB_worryF ~ avg_dep ") 
    formulas.append("avg_dep ~ WEC_sadF ") 
    formulas.append("avg_anx ~ WEB_worryF ") 
    formulas.append("avg_anx ~ WEC_sadF " )
    formulas.append("avg_dep ~ WEB_worryF ") 
    
    for formula in formulas:
      print('\t\tCounty Centered OLS on',formula)
      mod = smf.ols(formula, data=county_centered_data).fit()
      #print(mod.summary().tables[0])
      print(mod.summary().tables[1])


    # Run Mixed Linear Model
    if "region_name" in list(data.columns):
      for formula in formulas[:1]:
        md = smf.mixedlm(formula=formula, data=data, groups=data["region_name"])
        mdf = md.fit()
        print('\n',mdf.summary())


    # Run correlations (can be done on any level)
    group_on = 'yearweek_cnty' # yearweek = week x national, yearweek_cnty = week x county, yearweek_state = week x state, cnty = year x county
    merge = data.groupby(group_on).mean().reset_index()
    print("\nAggregated Data for Correlation\n",merge.head(5),'\n...')
    corr = merge.corr(method="pearson")
    print("\nLBA vs Gallup COVID by",group_on,'\n', corr)
    print(len(merge),"samples used for correlation")
    corr_plot = sns.heatmap(corr.head(2), center=0, square=True, linewidths=.5, annot=True)
    corr_plot.figure.savefig("LBA vs Gallup County Centered Data.png", bbox_inches='tight')


    # Make Gallup into feat table: ['group_id', 'feat', 'value', 'group_norm']
    # gallup_worry = gallup.copy(deep=True)
    # gallup_worry['feat'] = "WEB_worryF"
    # gallup_worry['value'] = gallup_worry['WEB_worryF']
    # gallup_sad = gallup.copy(deep=True)
    # gallup_sad['feat'] = "WEC_sadF"
    # gallup_sad['value'] = gallup_worry['WEC_sadF']
    # gallup_mysql = pd.concat([gallup_worry, gallup_sad])
    # gallup_mysql = gallup_mysql.rename(columns={"yearweek_cnty":"group_id"})
    # gallup_mysql["group_norm"] = gallup_mysql["value"]
    # gallup_mysql = gallup_mysql[['group_id', 'feat', 'value', 'group_norm']]
    # print(gallup_mysql)

    # from sqlalchemy import create_engine
    # from sqlalchemy.engine.url import URL
    # myDB = URL(drivername='mysql', host='localhost',
    #   database='ctlb2', query={ 'read_default_file' : "~/.my.cnf" }
    # )
    # engine = create_engine(name_or_url=myDB)

    # gallup_mysql.to_sql("feat$gallup_covid$yw_cnty",con=engine, if_exists='replace', index=False)