# Make sure to `conda activate`

from urllib.request import urlopen
import json
import plotly.express as px
from pymysql import connect
import datetime

gft=50

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

    import pandas as pd

    # tables = ["ctlb2.feat$dd_depAnxAng$timelines2019$yw_cnty$1gra", "ctlb2.feat$dd_depAnxAng$timelines2020$yw_cnty$1gra"]
    tables = ["ctlb2.feat$dd_depAnxLex_ctlb2$timelines19to20$05sc{}user$yw_cnty".format(gft)] 
    feat_value = "DEP_SCORE" # ANX_SCORE, DEP_SCORE
    filter = "WHERE feat = '{}'".format(feat_value)
    relevant_columns = "*"
    database = 'ctlb2'

    # Open default connection
    print('Connecting to MySQL...')
    connection  = connect(read_default_file="~/.my.cnf")

    print("Loading CTLB2")
    sql = "SELECT {} FROM {} {}".format( 
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
        df['date'] = df['yearweek'].apply(lambda yw: yearweek_to_dates(yw)[1])

    print(df);

    print("Group By county")
    grouped = df.groupby("cnty").mean().reset_index()
    print(grouped)

    # print("Collecting Super County Mapping")
    # super_cnty_mapping = pd.read_csv("~/super_counties/cnty_supes_mapping_{}.csv".format(gft),\
    #     dtype={'cnty':str, 'cnty_w_sups{}'.format(gft):str})
    # super_cnty_mapping['is_super'] = 0
    # super_cnty_mapping.loc[super_cnty_mapping['weight']<1,'is_super'] = 1
    # print(super_cnty_mapping.head())

    print("Loading GeoJSON")
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        string = response.read().decode('utf-8')
        counties = json.loads(string)

    print("Plotting",feat_value)
    fig = px.choropleth(df, geojson=counties, locations='cnty', color='group_norm',
                            color_continuous_scale="Blues", # Blues / Oranges
                            range_color=(min(grouped['group_norm']), max(grouped['group_norm'])),
                            scope="usa", # What region is plotted
                            labels={'group_norm':feat_value}
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Remove the legend
    fig.update_layout(coloraxis_showscale=False)

    print("Serving Choropleth")
    #fig.show()
    filename = "choropleth_" + feat_value + ".png"
    fig.write_image(filename)

    # print("Plotting Super Counties")
    # fig = px.choropleth(super_cnty_mapping, geojson=counties, locations='cnty', color='is_super',
    #                         color_continuous_scale="Bluered", # Blues / Oranges
    #                         range_color=(0, 1),
    #                         scope="usa", # What region is plotted
    #                         labels={'is_super':"Is in a super county?"}
    #                         )
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # fig.show()

if __name__ == "__main__":
    main()