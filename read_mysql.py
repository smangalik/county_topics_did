from pymysql import cursors, connect
import warnings
import numpy as np
import sys
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import json
import os.path

# Ignore warnings
warnings.catch_warnings()
warnings.simplefilter("ignore")

# How many of the top populous counties we want to keep
top_county_count = 100

# Number of topics studied
num_topics = 2000

# The county factors we want to cluster counties on
county_stats = "num_users,num_tweets"
county_factors_fields = "percent_male10, med_age10, log_med_house_income0509, high_school0509, bac0509"
county_factors_fields += ",log_pop_density10, percent_black10,percent_white10, foreign_born0509, rep_pre_2012, married0509"

# Number of nearest neighbors
k_neighbors = 30

# Diff in Diff Windows
event_timing_buffer = 4 # weeks before/after target event that are unaffected
diff_radius = 2 # distance in weeks from the middle of target event
diff_window_radius = 1 # how far to look around

# TODO define events on a per county basis
event_date_dict = {}

# <- 2013 [target_start][target_end] 2015 ->
# switch to a target start and target_end
# define a before and after length
# if a target_end is specified use that range as buffer
# if a target_end is NOT specified then target_end = target_set 
# the "after" will always contain target_end, or target_start if no target_end is specified 
# the "before" does not contain target_start


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

# Get nearest neighbors
def get_nearest_neighbors(county_factors_entry):
    pass

# Get category map from topic number to top words
def get_topic_map(cursor):
  topic_map = {}
  sql = 'select category, group_concat(term order by weight desc) as "terms" from dlatk_lexica.met_a30_2000_cp group by category;'
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
    sql = "select * from county_topic_change.feat$cat_met_a30_2000_cp_w$msgs_100u_{}$cnty$1gra;".format(table_year)
    cursor.execute(sql)

    for result in tqdm(cursor.fetchall_unbuffered()): # Read _unbuffered() to save memory
      _, county, topic_num, value, value_norm = result

      if topic_num == '_int': continue
      topic_num = int(topic_num)
      county = str(county).zfill(5)

      # Store county_topics
      # TODO replace table_year with yearweek
      if county_topics.get(county) is None:
        county_topics[county] = {}
      if county_topics[county].get(table_year) is None:
        county_topics[county][table_year] = [0] * num_topics
      county_topics[county][table_year][topic_num] = value_norm

  return county_topics

# Changes for new problems
def date_to_index(date, invert=False):
    # TODO extend to handle yearweeks
    if invert:
        return str(date)
    else:
        return int(date)

# TODO change how windowing is done to be based on a start and end
def avg_topic_usage(county,center_date,window_radius=diff_window_radius):

    # Determine indices of dates to check
    center_date = date_to_index(center_date)
    start_window = center_date - window_radius
    end_window = center_date + window_radius

    sum = np.zeros(num_topics)
    counter = 0
    for date in range(start_window,end_window+1):
        str_date = date_to_index(date,invert=True)
        if county_topics.get(county).get(str_date):
            topics_for_date = np.array(county_topics[county][str_date])
            sum = np.add(sum, topics_for_date)
            counter += 1
            # print("->",county,date,topics_for_date)

    if counter > 0:
        avg =  sum / counter
        # print("avg = ",avg)
        return avg
    else:
        print("No matching dates for", county,center_date,window_radius)
        return sum

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
    null_counties = {}
    for target in populous_counties:
        # TODO get the intervention timing for the county
        # target_event = county_events[target]
        target_event = '2014' # TODO remove hardcoded

        # Get the k top neighbors
        county_index = list(populous_counties).index(target)
        n_neighbors = neighbors.kneighbors([county_factors[county_index]], k_neighbors + 1, return_distance=False)
        null_counties[target] = []
        for i, n in enumerate(n_neighbors[0][1:]): # skip 0th entry (self)
            ith_closest_county = populous_counties[n]

            # TODO filter out counties with county events close by in time
            # if abs(county_events[ith_closest_county] - target_event) < event_timing_buffer: continue

            null_counties[target].append(ith_closest_county)

    # Calculate diff in diffs
    target_diffs = {}
    null_diffs = {}
    assert(diff_radius <= 2 * diff_window_radius)
    for target in populous_counties:
        # TODO get the intervention timing for the county
        # target_event = county_events[target]
        target_event = '2014' # TODO remove hardcoded
        target_event = date_to_index(target_event)

        before_target_event = target_event - diff_radius
        after_target_event = target_event + diff_radius

        target_before = avg_topic_usage(target, before_target_event)
        target_after = avg_topic_usage(target, after_target_event)
        target_diff = np.subtract(target_after,target_before)

        target_diffs[target] = target_diff

        null_counties_considered = null_counties[target]
        avg_null_diff = np.zeros(num_topics)
        for null_county in null_counties_considered:
            null_before = avg_topic_usage(null_county, before_target_event)
            null_after = avg_topic_usage(null_county, after_target_event)
            null_diff = np.subtract(null_after,null_before)

            # Add all differences, then divide by num of considered counties
            avg_null_diff = np.add(avg_null_diff, null_diff)

            # TODO Capture the standard deviation on top of averages 
            # the distribution of diffs should be a normal diffs

        # Average change from all null counties
        # TODO change the code to a list so we can get all stats
        null_diffs[target] = avg_null_diff / len(null_counties_considered)

        print('target_diffs',target_diffs)
        print('null_diffs',null_diffs)

        # TODO compare changes in avg_matched_counties with the changes in the target_county

        break # TODO only for testing

    # TODO Correlate these changes with life satisfaction or other metric



