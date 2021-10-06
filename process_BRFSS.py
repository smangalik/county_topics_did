import pandas as pd
import us
from datetime import datetime
import numpy as np

def date_to_yearweek(d):
  year, weeknumber, weekday = d.date().isocalendar()
  return str(year) + "_" + str(weeknumber)

# FIPs to ABBR
fips_abbr_map = us.states.mapping('fips', 'abbr')

# Read in the data
df = pd.read_sas("LLCP2019.XPT_",format="xport")

# Clean state
df['STATE_FIPS'] = df['_STATE'].apply(lambda x: fips_abbr_map[str(int(x)).zfill(2)])

# Get the date of the response
df['DATE_STR'] = df['IDATE'].str.decode("utf-8")
df['DATE'] = df['DATE_STR'].apply(lambda x: datetime.strptime(x, "%m%d%Y"))
df['YEARWEEK'] = df['DATE'].apply(lambda x: date_to_yearweek(x))

# Correct columns
df['MENTHLTH'] = df['MENTHLTH'].replace(88,0).replace([77,99,'BLANK'],np.nan)
df['POORHLTH'] = df['POORHLTH'].replace(88,0).replace([77,99,'BLANK'],np.nan)
df['ACEDEPRS'] = df['ACEDEPRS'].replace(2,0).replace([7,9,'BLANK'],np.nan)
df['_MENT14D'] = df['_MENT14D'].replace(9,np.nan)

# We care about these columns
rel_cols = ['STATE_FIPS','_STATE','DATE','YEARWEEK','MENTHLTH','POORHLTH','ACEDEPRS','_MENT14D']
mental_health = df[rel_cols]

mental_health.to_csv("BRFSS_mental_health.csv",index=False)
print("Output CSV")