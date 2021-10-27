# Intended to run on feature tables with `group_id` of format "yearweek:county"

from pymysql import cursors, connect
import pandas as pd
import numpy as np
import random, sys

d_threshold = 0.1

def splithalfs(x):
    shuffled = random.sample(x, len(x))
    split_index = len(shuffled)//2
    return shuffled[:split_index], shuffled[split_index:]

def permutation_cohens_d(x, n_perms=30):
    if len(x) <= 2: return 1.0, np.std(x)/np.sqrt(len(x))
    ds = []
    std = np.std(x)
    random.seed(a=25, version=2)
    for i in range(n_perms):
        shuffled = random.sample(x, len(x))
        split_index = len(shuffled)//2
        a,b = shuffled[:split_index], shuffled[split_index:]
        d = abs(np.mean(a) - np.mean(b)) / std
        ds.append(d)
    avg_d = np.mean(ds)
    d_stderr = np.std(ds) / np.sqrt(len(ds))   
    return avg_d, d_stderr

def cohens_d(x):
    if len(x) <= 2: return 1.0
    random.seed(a=25, version=2)
    shuffled = random.sample(x, len(x))
    split_index = len(shuffled)//2
    a,b = shuffled[:split_index], shuffled[split_index:]
    d = abs(np.mean(a) - np.mean(b)) / np.std(x)
    return d


def main():

    # Open default connection
    print('Connecting to MySQL...')
    connection  = connect(read_default_file="~/.my.cnf")

    # NOT INDIVIDUAL RESULTS
    tables = ["ctlb2.feat$dd_depAnxLex$timelines2019$yw_cnty$1gra",
              "ctlb2.feat$dd_depAnxLex$timelines2020$yw_cnty$1gra"]
    feat_val_col = "group_norm"
    groupby_col = "yearweek" # cnty, yearweek
    feat_value = "DEP_SCORE" # ANX_SCORE, DEP_SCORE
    filter = "WHERE feat = '{}'".format(feat_value)
    relevant_columns = "*"
    database = 'ctlb2'
    

    # tables = ["household_pulse.pulse"]
    # filter = ""
    # feat_val_col = "gad2_sum"
    # groupby_col = "msa_week" # EST_MSA, state, WEEK, msa_week, state_week
    # feat_value = "gad2_sum" # phq2_sum
    # relevant_columns = ",".join(['WEEK','state','EST_ST','EST_MSA','gad2_sum','phq2_sum'])
    # database = 'household_pulse'


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
    # grouped = grouped[grouped[feat_value].str.len() > 2] # list has more than 2 values

    # Calculate Cohen's D
    grouped['d'] = [cohens_d(x) for x in grouped[feat_value].values]
    
    # Calculate Permutation Test of D
    grouped['d_perm'] = [permutation_cohens_d(x) for x in grouped[feat_value].values]    
    grouped[["d_perm","d_perm_stderr"]] = grouped.apply(lambda x: permutation_cohens_d(x[feat_value]), axis=1,result_type ='expand') 

    # Filter D Values
    #significant = grouped[grouped['d_perm'] < d_threshold]
    significant = grouped[grouped['d_perm'] + grouped['d_perm_stderr'] * 1.96 < d_threshold]


    print("\nFor {} and {} we find the following {}/{} groups to have a Permutation Cohen's D less than {}:\n".format(
        feat_value, groupby_col, len(significant), len(grouped), d_threshold))
    print(significant[[groupby_col]+['d']+['d_perm']+['d_perm_stderr']])

    print("\nHere are all our findings:\n",grouped)
    

if __name__ == "__main__":
    main()