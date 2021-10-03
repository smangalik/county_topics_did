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
    feat_value = "DEP_SCORE"

    d_threshold = 0.1

    sql = "SELECT * FROM {} WHERE feat = '{}'".format(
        tables[0], feat_value
    )
    df = pd.read_sql(sql, connection)

    for table in tables[1:]:
        sql = "SELECT * FROM {} WHERE feat = '{}'".format(
            table, feat_value
        )
        df = pd.read_sql(sql, connection).append(df, ignore_index=True)
    
    
    print("Cleaning up columns")
    df = df[~df['group_id'].str.startswith((':'))]
    df[['yearweek','cnty']] = df['group_id'].str.split(":",expand=True,)
    
    print(df.head(25))

    grouped = df.groupby('cnty')['value'].apply(list).reset_index(name=feat_value)
    #df1['std'] = pd.DataFrame(df1[feat_value].values.tolist()).std(1)
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

    print("\nFor {} we find the following groups to have a Cohen's D less than {}:\n".format(feat_value,d_threshold))
    print(significant[['cnty','d']])
    print(grouped)
    

if __name__ == "__main__":
    main()