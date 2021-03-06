# run as `python3 read_mysql.py`

from pymysql import cursors, connect
import warnings
import numpy as np
import sys
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import json
import os.path
import matplotlib.pyplot as plt
import datetime

# Ignore warnings
warnings.catch_warnings()
warnings.simplefilter("ignore")

# How many of the top populous counties we want to keep
top_county_count = 300

# Number of topics studied
num_topics = 2000

# The county factors we want to cluster counties on
county_stats = "num_users,num_tweets"
county_factors_fields = "percent_male10, med_age10, log_med_house_income0509, high_school0509, bac0509"
county_factors_fields += ",log_pop_density10, percent_black10,percent_white10, foreign_born0509, rep_pre_2012, married0509"

# Number of nearest neighbors
k_neighbors = 30

# Diff in Diff Windows
default_before_start_window = 2 # additional time periods to consider before event start
default_after_end_window = 2 # additional time periods to consider after event end

# TODO event_date_dict[county] = [event_start, event_end]
county_events = {}

print('Connecting to MySQL...')

# Open default connection
connection  = connect(read_default_file="~/.my.cnf")

# Get the 100 most populous counties
def get_populous_counties(cursor, base_year):
  populous_counties = []
  # sql = "select * from county_topic_change.msgs_100u_{} order by num_users desc limit {};".format(base_year,top_county_count)
  sql = "select * from ctlb.counties{} order by num_of_users desc limit {};".format(base_year,top_county_count)
  cursor.execute(sql)
  for result in cursor.fetchall():
    cnty, num_users = result
    cnty = str(cnty).zfill(5)
    populous_counties.append(cnty)
  return populous_counties

# Get county features like [age, income, education, (race, ethnicity, well_being)]
def get_county_factors(cursor, base_year, relevant_counties, normalize=False):
  county_factors_h = len(relevant_counties)
  county_factors_w = len(county_stats.split(',')) + len(county_factors_fields.split(','))
  county_factors = np.zeros((county_factors_h,county_factors_w))

  for i, cnty in enumerate(relevant_counties):
      # Get stats
      # TODO change to CTLB
      sql = "select * from county_topic_change.msgs_100u_{} where cnty = {};".format(base_year,cnty)
      cursor.execute(sql)
      result = cursor.fetchone()
      if result is None:
          continue
      cnty, num_users, num_tweets = result
      county_factors[i][0] = np.log(float(num_users)) # log number of users
      county_factors[i][1] = np.log(float(num_tweets)) # log number of tweets

      # Get factors
      sql = "select {} from county_disease.county_PESH where cnty = {};".format(county_factors_fields,cnty)
      cursor.execute(sql)
      result = cursor.fetchone()
      factors = np.asarray(result,dtype=np.float32)
      county_factors[i][2:] = factors

  non_nan_counties = ~np.isnan(county_factors).any(axis=1)
  print("\nDropping counties with NaNs:",np.asarray(relevant_counties)[~non_nan_counties])
  county_factors = county_factors[non_nan_counties]
  relevant_counties = np.asarray(relevant_counties)[non_nan_counties]

  # Normalize columns of county_factors via z-score
  if normalize:
      county_factors = stats.zscore(county_factors, axis=0)

  # Fit the nearest neighbors
  neighbors = NearestNeighbors(n_neighbors=k_neighbors)
  neighbors.fit(county_factors)

  return county_factors, neighbors, relevant_counties

# Get category map from topic number to top words
def get_topic_map(cursor):
  topic_map = {}
  sql = 'select category, group_concat(term order by weight desc) as "terms" from dlatk_lexica.met_a30_2000_freq_t50ll group by category;'
  cursor.execute(sql)
  for result in cursor.fetchall():
    category, terms = result
    topic_map[category] = terms.split(',')[:10]
  return topic_map

# Iterate over all time units to create county_topics[county][year_week] = topics
def get_county_topics(cursor, table_years, relevant_counties):
  county_topics = {}
  for table_year in table_years:
    print('Processing {}'.format(table_year))

    # TODO change to CTLB
    sql = "select * from feat$cat_met_a30_2000_cp_w$timelines{}$yw_cnty$1gra;".format(table_year)
    sql = "select * from county_topic_change.feat$cat_met_a30_2000_cp_w$msgs_100u_{}$cnty$1gra;".format(table_year)

    cursor.execute(sql)

    for result in tqdm(cursor.fetchall_unbuffered()): # Read _unbuffered() to save memory
      _, county, topic_num, value, value_norm = result

      if topic_num == '_int': continue
      topic_num = int(topic_num)
      county = str(county).zfill(5)

      # Store county_topics
      # TODO replace table_year with yearweek, from splitting the county_yw
      if county_topics.get(county) is None:
        county_topics[county] = {}
      if county_topics[county].get(table_year) is None:
        county_topics[county][table_year] = [0] * num_topics
      county_topics[county][table_year][topic_num] = value_norm

  return county_topics

def date_to_yearweek(d):
  year, weeknumber, weekday = d.date().isocalendar()
  return str(year) + "_" + str(weeknumber)

# Changes for new problems
def date_to_index(date, invert=False):
    # TODO modify to handle datetime to yearweek indices
    if invert:
        return str(date)
    else:
        return int(date)

def avg_topic_from_dates(county,dates):
  topic_usages = []

  for date in dates:
    str_date = date_to_index(date,invert=True)
    if county_topics.get(county).get(str_date):
      topics_for_date = np.array(county_topics[county][str_date])
      topic_usages.append(topics_for_date)

  if len(topic_usages) == 0:
    print("No matching dates for", county, "on dates", dates)

  return np.mean(topic_usages, axis=0)

def topic_usage_before_and_after(county, event_start, event_end=None, before_start_window=default_before_start_window, after_start_window=default_after_end_window):

  # Parse start and end of event
  if event_end == None:
    event_end = event_start
  if not isinstance(event_start, int):
    event_start = date_to_index(event_start)
  if not isinstance(event_end, int):
    event_end = date_to_index(event_end)

  # Before window dates
  before_dates = list(range(event_start - before_start_window - 1, event_start))
  #print('before',before_dates)

  # After window dates
  after_dates = list(range(event_end, event_end + after_start_window + 1))
  #print('after',after_dates)

  # Get average usage
  return avg_topic_from_dates(county, before_dates), avg_topic_from_dates(county, after_dates), before_dates, after_dates


# Read in data with cursor
with connection:
  with connection.cursor(cursors.SSCursor) as cursor:
    print('Connected to',connection.host)

    # Determine the relevant counties
    base_year = 2019
    populous_counties = get_populous_counties(cursor, base_year)
    print("\nCounties with the most users in {}".format(base_year),populous_counties)

    # Create county_factor matrix and n-neighbors mdoel
    county_factors, neighbors, populous_counties = get_county_factors(cursor, 2016, populous_counties)
    print('\nCounty factor matrix shape:',county_factors.shape)

    # Display nearest neighbors for first county
    test_county = '36103' # Suffolk, NY
    test_county = '11001' # Washington, DC
    test_county_index = list(populous_counties).index(test_county)
    print('\nTest county (',test_county,')\n',county_factors[test_county_index])
    dist, n_neighbors = neighbors.kneighbors([county_factors[test_county_index]], 6, return_distance=True)
    for i, n in enumerate(n_neighbors[0]):
        print('#{}'.format(i),'Nearest County is',populous_counties[n],'with distance',dist[0][i])

    # Map topics to their key words
    topic_map = get_topic_map(cursor)
    print()
    print('Topic 0    =',topic_map['0'])
    print('Topic 344  =',topic_map['344'])
    print('Topic 160  =',topic_map['160'])
    print('Topic 1999 =',topic_map['1999'],'\n')

    # Get county topic information
    county_topics_json = "county_topics.json"

    if not os.path.isfile(county_topics_json):
        table_years = list(range(2012, 2017))
        county_topics = get_county_topics(cursor,table_years,populous_counties)
        with open(county_topics_json,"w") as json_file: json.dump(county_topics,json_file)
    print("Importing produced county topics")
    with open(county_topics_json) as json_file: county_topics = json.load(json_file)

    print("county_topics['06077']['2012'] =",county_topics['06077']['2012'][:5],'\n')
    print("county_topics['06077']['2013'] =",county_topics['06077']['2013'][:5],'\n')
    print("county_topics['06077']['2014'] =",county_topics['06077']['2014'][:5],'\n')
    print("county_topics['06077']['2015'] =",county_topics['06077']['2015'][:5],'\n')
    print("county_topics['06077']['2016'] =",county_topics['06077']['2016'][:5],'\n')

    # Get the closest k_neighbors for each populous_county we want to examine
    county_representation = {}

    matched_counties = {}
    for target in populous_counties:
        target_event_start,target_event_end = county_events.get(target,[None,None])
        target_event = '2014' # TODO remove hardcoded

        # Get the k top neighbors
        county_index = list(populous_counties).index(target)
        n_neighbors = neighbors.kneighbors([county_factors[county_index]], k_neighbors + 1, return_distance=False)
        matched_counties[target] = []
        for i, n in enumerate(n_neighbors[0][1:]): # skip 0th entry (self)
            ith_closest_county = populous_counties[n]

            # determine how much each county appears
            if ith_closest_county not in county_representation.keys():
              county_representation[ith_closest_county] = 0
            county_representation[ith_closest_county] += 1

            # TODO filter out counties with county events close by in time
            ith_closest_county_event, _ = county_events.get(ith_closest_county,[None,None])
            # if abs(ith_closest_county_event - target_event) < event_timing_buffer: continue

            matched_counties[target].append(ith_closest_county)

    neighbor_counts = sorted(county_representation.items(), key=lambda kv: kv[1])
    print("\nCount of times _ is a neighbor\n", neighbor_counts[:10],"...",neighbor_counts[-10:], '\n')

    # Calculate diff in diffs
    target_diffs = {}
    matched_diffs = {}
    matched_befores = {}
    for target in populous_counties:
        target_event_start,target_event_end = county_events.get(target,[None,None])
        target_event_start,target_event_end = '2014','2015' # TODO remove hardcoded

        if target_event_start is None and target_event_end is None:
          continue

        target_before, target_after, dates_before, dates_after = topic_usage_before_and_after(target, event_start=target_event_start, event_end=target_event_end)
        target_diff = np.subtract(target_after,target_before)

        target_diffs[target] = target_diff

        matched_counties_considered = matched_counties[target]
        matched_diffs[target] = []
        matched_befores[target] = []
        for matched_county in matched_counties_considered:
            matched_before, matched_after, _, _ = topic_usage_before_and_after(matched_county, event_start=target_event_start, event_end=target_event_end)
            matched_diff = np.subtract(matched_after,matched_before)

            # Add all differences, then divide by num of considered counties
            matched_diffs[target].append(matched_diff)
            matched_befores[target].append(matched_before)

        # print('target_diffs',target_diffs)
        # print('matched_diffs',matched_diffs)

        # Average/std change from all matched counties
        avg_matched_diff = np.mean(matched_diffs[target], axis=0)
        std_matched_diff = np.std(matched_diffs[target], axis=0)
        # print("\nAverage change in matched counties:", avg_matched_diff)
        # print("std. dev. change in matched counties:", avg_matched_diff)

        # Diff in Diff
        target_expected = target_before + avg_matched_diff
        intervention_effects = target_after - target_expected

        # compare changes in avg_matched_counties with the changes in the target_county
        # print()
        # print("Target Before:\n", target_before)
        # print("Target After (with intervention / observation)\n", target_after)
        # print("Target After (without intervention / expected):\n", target_expected)
        # print("Intervention Effect:\n", intervention_effects)

        # Plot findings
        for feature_num in range(10): # TODO run on ALL topics
          x = [datetime.datetime(2012, 1, 1), datetime.datetime(2016, 1, 1)] # TODO calc in between dates
          matches_before = np.array(matched_befores[target])
          matches_after = np.array(matched_befores[target]) + np.array(matched_diffs[target])
          avg_match_before = np.mean(matches_before[:,feature_num])
          avg_match_after = np.mean(matches_after[:,feature_num])
          std_match_before = np.std(matches_before[:,feature_num])
          std_match_after = np.std(matches_after[:,feature_num])
          stderr_match_before = std_match_before / np.sqrt(k_neighbors)
          stderr_match_after = std_match_after / np.sqrt(k_neighbors)

          is_significant = abs(intervention_effects[feature_num]) > stderr_match_after*1.96
          if not is_significant:
            continue
          increase_decrease = "increased" if intervention_effects[feature_num] > 0 else "decreased"

          print("Change in", topic_map[str(feature_num)][:8], increase_decrease, \
            "significantly -> Topic #", feature_num)

          ci_down = [target_before[feature_num]-stderr_match_before, target_expected[feature_num]-stderr_match_after]
          ci_up = [target_before[feature_num]+stderr_match_before, target_expected[feature_num]+stderr_match_after]
          ci_down_2 = [target_before[feature_num]-stderr_match_before*1.96, target_expected[feature_num]-stderr_match_after*1.96]
          ci_up_2 = [target_before[feature_num]+stderr_match_before*1.96, target_expected[feature_num]+stderr_match_after*1.96]

          plt.clf() # reset plot
          fig, ax = plt.subplots()
          fig.set_size_inches(5, 6)
          plt.plot(x, [target_before[feature_num], target_after[feature_num]], 'b-', label='Target (Actual)')
          plt.plot(x, [target_before[feature_num], target_expected[feature_num]],'c--', label='Target (Expected)')
          plt.plot([x[0]]*30, matches_before[:,feature_num], 'r+', alpha=0.2)
          plt.plot([x[1]]*30, matches_after[:,feature_num], 'r+', alpha=0.2)
          #plt.plot(x,[avg_match_before,avg_match_after],'r--',label='Average Match')
          plt.fill_between(x, ci_down, ci_up, color='c', alpha=0.3)
          plt.fill_between(x, ci_down_2, ci_up_2, color='c', alpha=0.2)
          plt.plot([x[1],x[1]], [target_after[feature_num], target_expected[feature_num]], 'k--', label='Intervention Effect')
          plt.title("Diff in Diff Analysis for County " + str(target))

          # Format plot
          plt.gcf().autofmt_xdate()
          #ax.set_xticks(x) # TODO indicate the entire windows being used for the before and after?
          ax.set_xticks([
            datetime.datetime(int(dates_before[0]), 1, 1),  # begin before
            datetime.datetime(int(dates_before[-1]), 1, 1), # end before
            datetime.datetime(int(dates_after[0]), 1, 1),   # begin after
            datetime.datetime(int(dates_after[-1]), 1, 1),  # end after
          ])
          plt.xlabel("Time".format(dates_before,dates_after))
          plt.ylabel(str(topic_map[str(feature_num)][:4]) + " Usage")
          plt.legend()
          plt.tight_layout()

          plt_name = "did_topic{}_cnty{}_time{}{}{}-{}{}{}.png".format( \
            feature_num,target,x[0].year,x[0].month,x[0].day,x[1].year,x[0].month,x[0].day)

          plt.savefig(plt_name)

        # plt.show()

        break # TODO remove when done testing

    # TODO Correlate these changes with life satisfaction or other metric

    print()
