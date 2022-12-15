# Returns the first and last day of a week given as "2020_11"
import datetime

def date_to_yearweek(d:datetime):
  year, weeknumber, weekday = d.date().isocalendar()
  return str(year) + "_" + str(weeknumber)

def yearweek_to_dates(yw:str):
  year, week = yw.split("_")
  year, week = int(year), int(week)

  first = datetime.datetime(year, 1, 1)
  base = 1 if first.isocalendar()[1] == 1 else 8
  monday = first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))
  sunday = monday + datetime.timedelta(days=6)
  thursday = monday + datetime.timedelta(days=3)
  return monday, thursday, sunday

def date_to_quarter(dt):
    return "Q{}".format( (dt.month+2) // 3 )