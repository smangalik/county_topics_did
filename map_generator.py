from urllib.request import urlopen
import json
import plotly.express as px
from pymysql import connect
import datetime

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

    tables = ["ctlb2.feat$dd_depAnxAng$timelines2019$yw_cnty$1gra",
            "ctlb2.feat$dd_depAnxAng$timelines2020$yw_cnty$1gra"]
    feat_value = "DEP_SCORE" # ANX_SCORE, DEP_SCORE, ANG_SCORE
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

    print("Loading GeoJSON")
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        string = response.read().decode('utf-8')
        counties = json.loads(string)

    # print("Loading Data")
    # import pandas as pd
    # df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
    #                 dtype={"fips": str})

    print("Plotting")
    fig = px.choropleth(df, geojson=counties, locations='cnty', color='group_norm',
                            color_continuous_scale="Viridis",
                            range_color=(min(grouped['group_norm']), max(grouped['group_norm'])),
                            scope="usa", # What region is plotted
                            labels={'group_norm':feat_value}
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Remove the legend
    # fig.update_layout(coloraxis_showscale=False)

    print("Serving Choropleth")
    #fig.show()
    filename = "choropleth_" + feat_value + ".png"
    fig.write_image(filename)

if __name__ == "__main__":
    main()