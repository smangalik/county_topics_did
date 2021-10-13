# Intended to run on feature tables with `group_id` of format "yearweek:county"

from pymysql import cursors, connect
import pandas as pd
import numpy as np
import random

def calculate_splithalfs(county, person_id, score):
    d_map = {} # d_map[group] = d
    for group in group_column:
        #std = standard dev of score in county
        #half1, half2 = random_cut (permutation -> split in half)
        #d_map[group] = abs(half1.mean() - half2.mean()) / std
        pass
    return d_map 

def splithalfs(x):
    shuffled = random.sample(x, len(x))
    split_index = len(shuffled)//2
    return shuffled[:split_index], shuffled[split_index:]

def main():

    # Open default connection
    print('Connecting to MySQL...')
    connection  = connect(read_default_file="~/.my.cnf")

    tables = ["ctlb2.feat$dd_depAnxLex$timelines2019$yw_cnty$1gra",
              "ctlb2.feat$dd_depAnxLex$timelines2020$yw_cnty$1gra"]
    feat_val_col = "value"
    groupby_col = "cnty"
    feat_value = "DEP_SCORE"
    filter = "WHERE feat = '{}'".format(feat_value)
    relevant_columns = "*"
    database = 'ctlb2'
    

    # tables = ["household_pulse.pulse"]
    # filter = ""
    # feat_val_col = "gad2_sum"
    # groupby_col = "state_week"
    # feat_value = "gad2_sum"
    # relevant_columns = ",".join(['WEEK','state','EST_ST','EST_MSA','gad2_sum','phq2_sum'])
    # database = 'household_pulse'

    d_threshold = 0.1

    sql = "SELECT {} FROM {} {}".format( # TODO remove limit
        relevant_columns, tables[0], filter
    )
    df = pd.read_sql(sql, connection)

    for table in tables[1:]:
        sql = "SELECT {} FROM {} {}".format(
            relevant_columns, table, filter
        )
        df = pd.read_sql(sql, connection).append(df, ignore_index=True)
    
    
    print("Cleaning up columns")
    if database == 'ctlb2':
        df = df[~df['group_id'].str.startswith((':'))]
        df[['yearweek','cnty']] = df['group_id'].str.split(":",expand=True,)

    if database == 'household_pulse':
        df['state_week'] = df['state'] + "_" + df['WEEK'].map(str)
        df['msa_week'] = df['EST_MSA'] + "_" + df['WEEK'].map(str)
    
    print(df.head(25))

    grouped = df.groupby(groupby_col)[feat_val_col].apply(list).reset_index(name=feat_value)
    grouped[feat_value] = grouped[feat_value].apply(lambda x: [i for i in x if str(i) != "nan"]) # clean NaNs
    grouped['std'] = [np.array(x).std() for x in grouped[feat_value].values]
    grouped = grouped[grouped[feat_value].str.len() > 1] # list has more than 1 value

    # Shuffle split into halves
    grouped[["half1","half2"]] = grouped.apply(lambda x: splithalfs(x[feat_value]), axis=1,result_type ='expand') 
    grouped = grouped.drop([feat_value], axis=1)
    grouped['mean1'] = [np.array(x).mean() for x in grouped['half1'].values]
    grouped['mean2'] = [np.array(x).mean() for x in grouped['half2'].values]

    # Calculate Cohen's D
    grouped['d'] = abs(grouped['mean1'] - grouped['mean2']) / grouped['std']

    # Filter D Values
    significant = grouped[grouped['d'] < d_threshold]

    print("\nFor {} we find the following {} groups to have a Cohen's D less than {}:\n".format(feat_value,len(significant),d_threshold))
    print(significant[[groupby_col]+['d']])

    print("\nHere are all our findings:\n")
    print(grouped)
    

if __name__ == "__main__":
    main()