from pymysql import cursors, connect
import pandas as pd

def main():

    # Open default connection
    print('Connecting to MySQL...')
    connection  = connect(read_default_file="~/.my.cnf")

    # With the 1upt version of the data
    tables = ["ctlb2.feat$dd_depAnxAng_rw$timelines2019$1upt100user$year_cnty",
              "ctlb2.feat$dd_depAnxAng_rw$timelines2020$1upt100user$year_cnty"]

    sql = "SELECT * FROM {} ".format( tables[0] )
    df = pd.read_sql(sql, connection)

    for table in tables[1:]:
        sql = "SELECT * FROM {}".format( table )
        df = pd.read_sql(sql, connection).append(df, ignore_index=True)


    print("Cleaning up columns")
    df = df[~df['group_id'].str.startswith((':'))]
    df[['year','cnty']] = df['group_id'].str.split(":",expand=True,)
    df['cnty_feat'] = df['cnty'] + ":" + df['feat']

    # Aggregate to county
    grouped = df
    # grouped = df.groupby(by="cnty_feat").agg({"value":"mean", "group_norm":"mean", "yearweek":"nunique"}).reset_index()
    # grouped[['cnty','feat']] = grouped['cnty_feat'].str.split(":",expand=True,)

    # At least n yearweeks must be present
    # n = 10
    # grouped = grouped[grouped['yearweek'] > n]

    print(grouped)

    final = grouped[['group_id','feat','value','group_norm','cnty','year']]
    final = final.rename(columns={"cnty_year":"group_id"})
    final.insert(0, 'id', range(1, 1 + len(final)))


    print(final)

    # Output CSV
    final.to_csv("./data/feat.dd_depAnxAng_rw$timelines$1upt100user$year_cnty.csv", index=False)

if __name__ == "__main__":
    main()
