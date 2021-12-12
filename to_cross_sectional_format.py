from pymysql import cursors, connect
import pandas as pd

def main():

    # Open default connection
    print('Connecting to MySQL...')
    connection  = connect(read_default_file="~/.my.cnf")

    # TODO redo with the 1upt version of the data
    tables = ["ctlb2.feat$dd_depAnxAng$timelines2019$yw_cnty$1gra",
              "ctlb2.feat$dd_depAnxAng$timelines2020$yw_cnty$1gra"]
    database = 'ctlb2'

    sql = "SELECT * FROM {} ".format( tables[0] )
    df = pd.read_sql(sql, connection)

    for table in tables[1:]:
        sql = "SELECT * FROM {}".format( table )
        df = pd.read_sql(sql, connection).append(df, ignore_index=True)
    
    
    print("Cleaning up columns")
    df = df[~df['group_id'].str.startswith((':'))]
    df[['yearweek','cnty']] = df['group_id'].str.split(":",expand=True,)
    df['cnty_feat'] = df['cnty'] + ":" + df['feat'] 

    # Aggregate to county
    grouped = df.groupby(by="cnty_feat").agg({"value":"mean", "group_norm":"mean", "yearweek":"nunique"}).reset_index()
    grouped[['cnty','feat']] = grouped['cnty_feat'].str.split(":",expand=True,)

    # At least n yearweeks must be present
    n = 10
    grouped = grouped[grouped['yearweek'] > n]

    print(grouped)

    final = grouped[['cnty','feat','value','group_norm']]
    final = final.rename(columns={"cnty":"group_id"})
    final.insert(0, 'id', range(1, 1 + len(final)))
    

    print(final)

    # Output CSV
    final.to_csv("./data/cross_sectional_dd_depAnxAng_3upts.csv", index=False)

if __name__ == "__main__":
    main()