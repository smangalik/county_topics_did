# Intended to run on feature tables with `group_id` of format "yearweek:county"

from pymysql import cursors, connect
import pandas as pd
import numpy as np
import datetime

GFT = 50

def yearweek_to_dates(yw):
  year, week = yw.split("_")
  year, week = int(year), int(week)

  first = datetime.datetime(year, 1, 1)
  base = 1 if first.isocalendar()[1] == 1 else 8
  monday = first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))
  sunday = monday + datetime.timedelta(days=6)
  thursday = monday + datetime.timedelta(days=3)
  return monday, thursday, sunday

def main():

    # Open default connection
    print('Connecting to MySQL...')
    connection  = connect(read_default_file="~/.my.cnf")

    # Get supplemental data
    county_info = pd.read_csv("county_fips_data.csv",encoding = "utf-8")
    county_info['cnty'] = county_info['fips'].astype(str).str.zfill(5)

    # tables = ["ctlb2.feat$dd_depAnxAng_rw$timelines2019$3upt3_user$yw_cnty$1gra",
    #           "ctlb2.feat$dd_depAnxAng_rw$timelines2020$3upt3_user$yw_cnty$1gra"]
    #tables = ["ctlb2.feat$dd_depAnxLex_ctlb2$timelines19to20$3upt50user$yw_cnty"] # 50 GFT, old scaling
    #tables = ["ctlb2.feat$dd_depAnxLex_ctlb2$timelines19to20$05sc200user$yw_cnty"] # 200 GFT, fixed scaling
    #tables = ["ctlb2.feat$dd_depAnxLex_ctlb2_nostd$timelines19to20_lex_3upts$yw_cnty"] # Pick your own GFT! 16to16
    tables = ["ctlb2.feat$dd_daa_c2adpt_ans_nos$timelines19to20_lex_3upts$yw_cnty"] # Pick your own GFT! 16to8
    table = tables[0]

    sql = "SELECT * FROM {} WHERE n_users >= {}".format( table, GFT )
    df = pd.read_sql(sql, connection)

    for table in tables[1:]:
        sql = "SELECT * FROM {} WHERE n_users >= {}".format( table, GFT )
        df = pd.read_sql(sql, connection).append(df, ignore_index=True)


    print("Cleaning up columns")
    df = df[~df['yearweek_cnty'].str.startswith((':'))]
    df[['yearweek','cnty']] = df['yearweek_cnty'].str.split(":",expand=True,)
    df['date'] = df['yearweek'].apply(lambda yw: yearweek_to_dates(yw)[1]) # Thursday

    # Merge in additional information
    df = pd.merge(df,county_info,on='cnty')
    df['county_name'] = df['county_name'].str.replace(" County","")
    df['county_name'] = df['county_name'].str.replace(" Municipality","")
    df['county_name'] = df['county_name'].str.replace(" City and Borough","")
    df['county_name'] = df['county_name'].str.replace(" Borough","")
    df['county_name'] = df['county_name'].str.replace(" city","")

    print("\nOriginal Data")
    print(df.head())

    df = df[['date','county_name','state_name','cnty','feat','wavg_score']]

    anx = df[df['feat']=="ANX_SCORE"].drop(['feat'], axis=1).rename(columns={"wavg_score":"ANX_SCORE"})
    dep = df[df['feat']=="DEP_SCORE"].drop(['feat'], axis=1).rename(columns={"wavg_score":"DEP_SCORE"})

    print("\nAnxiety then Depression Data")
    print(anx.head())
    print(dep.head())

    print("\nNY Times Format")
    nytimes = pd.merge(anx,dep,on=['date','county_name','state_name','cnty'], how='outer')
    nytimes.columns = ['date','county','state','fips','anxiety','depression']
    nytimes.sort_values('date', ascending=True, inplace=True) # sort by date
    print(nytimes.sample(10))
    print("\nJust Cook County")
    print(nytimes[nytimes['county']=="Cook"].head(10))
    print(nytimes[nytimes['county']=="Cook"].tail(10))

    # Write to CSV
    nytimes.to_csv("us-counties.csv", index=False)

if __name__ == "__main__":
    main()
