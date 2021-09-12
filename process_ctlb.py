""" 
run as `python3 process_ctlb.py` or `python3 process_ctlb.py --not-topics` 
"""

from pymysql import cursors, connect

import warnings
import sys
import argparse
from tqdm import tqdm
import json
import os.path
import datetime, time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


# Ignore warnings
warnings.catch_warnings()
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Process feature table for diff in diff analysis")
parser.add_argument('--not_topics', dest="topics", default=True ,action='store_false', help='Is the analysis done on topics?')
args = parser.parse_args()

# is the analysis being done on topics? (topic_num needs to be interpreted)
topics = args.topics

# The county factors we want to cluster counties on
county_stats = "num_users,num_tweets"
county_factors_fields = "percent_male10, med_age10, log_med_house_income0509, high_school0509, bac0509"
county_factors_fields += ",log_pop_density10, percent_black10,percent_white10, foreign_born0509, rep_pre_2012, married0509"

# Number of principal componenets used in PCA
pca_components = 5

# How many of the top populous counties we want to keep
top_county_count = 300 # 3232 is the maximum number, has a serious effect on results

# Number of features studied
num_feats = 2000

# Number of nearest neighbors
k_neighbors = 30

# Diff in Diff Windows
default_before_start_window = 1 # additional weeks to consider before event start
default_after_end_window = 1 # additional weeks to consider after event end
default_event_buffer = 1 # number of weeks to ignore before and after event

# Confidence Interval Multiplier
ci_window = 1.96

# Scale factor for y-axis
scale_factor = 100000

# event_date_dict[county] = [event_start (datetime, exclusive), event_end (datetime, inclusive), event_name]
county_events = {}
county_events['11000'] = [datetime.datetime(2020, 5, 25), datetime.datetime(2020, 6, 21), "Death of George Floyd"]
county_events['11001'] = [datetime.datetime(2020, 5, 25), None, "Death of George Floyd"]

# populate events from countyFirsts.csv
first_covid_case = {}
first_covid_death = {}
fips_to_name = {}
fips_to_population = {}
with open("/data/smangalik/countyFirsts.csv") as countyFirsts:
    lines = countyFirsts.read().splitlines()[1:] # read and skip header
    for line in lines:
      fips, county, state, population, firstCase, firstDeath = line.split(",")
      fips_to_name[fips] = county + ", " + state
      fips_to_population[fips] = int(population)
      first_covid_case[fips] = [datetime.datetime.strptime(firstCase, '%Y-%m-%d'),None,"First Covid Case"]
      if firstDeath != "":
        first_covid_death[fips] = [datetime.datetime.strptime(firstDeath, '%Y-%m-%d'),None,"First Covid Death"]

print('Connecting to MySQL...')

# Open default connection
connection  = connect(read_default_file="~/.my.cnf")

# Get the 100 most populous counties
def get_populous_counties(cursor, base_year):
  populous_counties = []
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
      # TODO change to CTLB to get better stats on user tweets [cnty|num_users|num_tweets]
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

  # Take the PCA of the county factors to calculate neighbors
  pca = make_pipeline(StandardScaler(), 
                      PCA(n_components=pca_components, random_state=0))
  pca.fit(county_factors)
  county_factors = pca.transform(county_factors)  

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

# Iterate over all time units to create county_feats[county][year_week] = feats
def get_county_feats(cursor, table_years):
  county_feats = {}
  for table_year in table_years:
    print('Processing {}'.format(table_year))

    if topics:
      sql = "select * from ctlb2.feat$cat_met_a30_2000_cp_w$timelines{}$yw_cnty$1gra;".format(table_year)
    else:
      sql = "select * from ctlb2.feat$dd_depAnxLex$timelines{}$yw_cnty$1gra;".format(table_year)
    cursor.execute(sql)

    for result in tqdm(cursor.fetchall_unbuffered()): # Read _unbuffered() to save memory
      yw_county, feat, value, value_norm = result

      if feat == '_int': continue
      yearweek, county = yw_county.split(":")
      if county == "" or yearweek == "": continue
      county = str(county).zfill(5)

      # Store county_feats
      if county_feats.get(county) is None:
        county_feats[county] = {}
      if county_feats[county].get(yearweek) is None:
        if topics:
          county_feats[county][yearweek] = np.zeros(num_feats)
        else:
          county_feats[county][yearweek] = {}
      county_feats[county][yearweek][feat] = value_norm

  return county_feats

# Returns the yearweek that the date is within
def date_to_yearweek(d):
  year, weeknumber, weekday = d.date().isocalendar()
  return str(year) + "_" + str(weeknumber)

# Returns the first and last day of a week given as "2020_11"
def yearweek_to_dates(yw):
  year, week = yw.split("_")
  year, week = int(year), int(week)

  first = datetime.datetime(year, 1, 1)
  base = 1 if first.isocalendar()[1] == 1 else 8
  monday = first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))
  sunday = monday + datetime.timedelta(days=6)
  return monday, sunday

# Take in a county and some yearweeks, then average their feat usage
def avg_topic_usage_from_dates(county,dates):
  feat_usages = []

  for date in dates:
    if county_feats.get(county) is not None and county_feats.get(county).get(date) is not None:
      feats_for_date = np.array(county_feats[county][date]) * scale_factor # scale values and store
      feat_usages.append(feats_for_date)

  if len(feat_usages) == 0:
    # print("No matching dates for", county, "on dates", dates)
    return None

  #print("feats_for_date",feats_for_date)
  #print("feat_usages",feat_usages)

  return np.mean(feat_usages, axis=0)

# TODO change to accomodate non-indexed feats; OR force all things to be indexed and have a map of values
def avg_feat_usage_from_dates(county,dates):
  feat_usages = []

  for date in dates:
    if county_feats.get(county) is not None and county_feats.get(county).get(date) is not None:
      feats_for_date = np.array(county_feats[county][date])
      feat_usages.append(feats_for_date)

  if len(feat_usages) == 0:
    # print("No matching dates for", county, "on dates", dates)
    return None

  print("feats_for_date",feats_for_date)
  print("feat_usages",feat_usages)

  return np.mean(feat_usages, axis=0)

def feat_usage_before_and_after(county, event_start, event_end=None, 
                                 before_start_window=default_before_start_window, 
                                 after_start_window=default_after_end_window,
                                 event_buffer=default_event_buffer):

  # If no event end specified, end = start
  if event_end == None:
    event_end = event_start 

  #print('center',date_to_yearweek(event_start))

  # Apply buffer
  event_start = event_start - datetime.timedelta(days=event_buffer*7)
  event_end = event_end + datetime.timedelta(days=event_buffer*7)

  #print('start',date_to_yearweek(event_start))
  #print('end',date_to_yearweek(event_end))

  # Before window dates, ex. '2020_11',2 -> ['2020_08', '2020_09', '2020_10']
  before_dates = []
  for i in range(1,before_start_window + 2):
    day = event_start - datetime.timedelta(days=i*7)
    before_dates.append(day)
  before_dates = [date_to_yearweek(x) for x in before_dates]
  before_dates = list(set(before_dates))
  before_dates.sort()
  #print('before',before_dates)

  # After window dates, ex. '2020_11',2 -> ['2020_11', '2020_12', '2020_13']
  after_dates = []
  for i in range(after_start_window + 1):
    day = event_end + datetime.timedelta(days=i*7)
    after_dates.append(day)
  after_dates = [date_to_yearweek(x) for x in after_dates]
  after_dates = list(set(after_dates))
  after_dates.sort()
  #print('after',after_dates)

  # Get average usage
  if topics:
    return avg_topic_usage_from_dates(county, before_dates), avg_topic_usage_from_dates(county, after_dates), before_dates, after_dates
  else:
    return avg_feat_usage_from_dates(county, before_dates), avg_feat_usage_from_dates(county, after_dates), before_dates, after_dates

def plot_diff_in_diff_per_county():

  if topics:
    list_features = range(num_feats)
  else:
    list_features = [] # TODO this will be a list of values

  for feature_num in list_features:
    matches_before = np.array(matched_befores[target])
    matches_after = np.array(matched_befores[target]) + np.array(matched_diffs[target])
    avg_match_before = np.mean(matches_before[:,feature_num])
    avg_match_after = np.mean(matches_after[:,feature_num])
    std_match_before = np.std(matches_before[:,feature_num])
    std_match_after = np.std(matches_after[:,feature_num])
    stderr_match_before = std_match_before / np.sqrt(k_neighbors)
    stderr_match_after = std_match_after / np.sqrt(k_neighbors)

    is_significant = abs(intervention_effects[feature_num]) > stderr_match_after*ci_window
    if not is_significant: 
      continue # only plot significant results

    # List significant changes
    increase_decrease = "increased" if intervention_effects[feature_num] > 0 else "decreased"
    if topics:
      print("Change in", topic_map[str(feature_num)][:8], increase_decrease, "significantly -> Topic #", feature_num)

    plt.clf() # reset plot

    # Confidence Intervals
    ci_down = [target_before[feature_num]-stderr_match_before, target_expected[feature_num]-stderr_match_after]
    ci_up = [target_before[feature_num]+stderr_match_before, target_expected[feature_num]+stderr_match_after]
    ci_down_2 = [target_before[feature_num]-stderr_match_before*ci_window, target_expected[feature_num]-stderr_match_after*ci_window]
    ci_up_2 = [target_before[feature_num]+stderr_match_before*ci_window, target_expected[feature_num]+stderr_match_after*ci_window]
    
    # Create Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    plt.plot(x, [target_before[feature_num], target_after[feature_num]], 'b-', label='Target (Actual)')
    plt.plot(x, [target_before[feature_num], target_expected[feature_num]],'c--', label='Target (Expected)')
    plt.plot([x[0]]*30, matches_before[:,feature_num], 'r+', alpha=0.2)
    plt.plot([x[1]]*30, matches_after[:,feature_num], 'r+', alpha=0.2)
    plt.plot(x,[avg_match_before,avg_match_after],'r--',label='Average Match')
    plt.fill_between(x, ci_down, ci_up, color='c', alpha=0.3)
    plt.fill_between(x, ci_down_2, ci_up_2, color='c', alpha=0.2)
    plt.plot([x[1],x[1]], [target_after[feature_num], target_expected[feature_num]], 'k--', \
      label='Intervention Effect ({})'.format(round(intervention_effects[feature_num],5)))
    plt.title("County " + str(target) + " before/after " + event_name)

    # Format plot
    plt.gcf().autofmt_xdate()
    ax.set_xticks(xticks)
    ax.set_xticklabels([
      "{} weeks before event".format(default_event_buffer + default_before_start_window + 1),
      "{} week before event".format(default_event_buffer),
      "{} week after event".format(default_event_buffer),
      "{} weeks after event".format(default_event_buffer + default_after_end_window + 1)
    ])
    plt.xlabel("Time".format(dates_before,dates_after))
    if topics:
      plt.ylabel(str(topic_map[str(feature_num)][:5]) + "\ntopic mentions per 100,000")
    else: 
      plt.ylabel(str(feature_num) + " Usage")
    plt.legend()
    plt.tight_layout()

    plt_name = "did_feat_{}_cnty{}_time{}{}{}-{}{}{}.png".format( \
      feature_num,target,x[0].year,x[0].month,x[0].day,x[1].year,x[0].month,x[0].day)

    plt.savefig(plt_name)

# Read in data with cursor
with connection:
  with connection.cursor(cursors.SSCursor) as cursor:
    print('Connected to',connection.host)

    print("topics",topics)

    # Determine the relevant counties
    base_year = 2019
    #populous_counties = get_populous_counties(cursor, base_year)
    #print("\nCounties with the most users in {}".format(base_year),populous_counties[:25],"...")
    populous_counties = sorted(fips_to_population, key=fips_to_population.get, reverse=True)[:top_county_count]
    print("\nCounties with the most users in 2021",populous_counties[:25],"...")
    

    # Create county_factor matrix and n-neighbors mdoel
    # TODO use better 2020 data in general
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
    if topics:
      topic_map = get_topic_map(cursor)
      print()
      print('Topic 0    =',topic_map['0'])
      print('Topic 344  =',topic_map['344'])
      print('Topic 160  =',topic_map['160'])
      print('Topic 1999 =',topic_map['1999'],'\n')


    # Get county feat information
    if topics:
      county_feats_json = "/data/smangalik/county_topics_ctlb.json"
    else: 
      county_feats_json = "/data/smangalik/county_feats_ctlb.json"
    if not os.path.isfile(county_feats_json):
        #table_years = list(range(2012, 2017))
        table_years = [2019,2020]
        county_feats = get_county_feats(cursor,table_years)
        with open(county_feats_json,"w") as json_file: json.dump(county_feats,json_file)
    start_time = time.time()
    print("Importing produced county features")
    with open(county_feats_json) as json_file: 
      county_feats = json.load(json_file)
    print("--- %s seconds to import ---" % (time.time() - start_time))

    if topics:
      print("county_feats['48117']['2020_19'] =",county_feats['48117']['2020_19'][:4],"...")
      print("county_feats['11001']['2020_19'] =",county_feats['11001']['2020_19'][:4],"...")
      print("county_feats['11001']['2020_20'] =",county_feats['11001']['2020_20'][:4],"...")
      print("county_feats['11001']['2020_21'] =",county_feats['11001']['2020_21'][:4],"...")
      print("county_feats['11001']['2020_25'] =",county_feats['11001']['2020_25'][:4],"...")
      print("county_feats['11001']['2020_26'] =",county_feats['11001']['2020_26'][:4],"...")
      print("county_feats['11001']['2020_27'] =",county_feats['11001']['2020_27'][:4],"...")
      available_yws = list(county_feats['11001'].keys())
      available_yws.sort()
      print("\nAvailable weeks for 11001:",  available_yws)
    else: 
      print("county_feats['11001']['2020_19']['DEP_SCORE'] =",county_feats['11001']['2020_19']['DEP_SCORE'])
      print("county_feats['11001']['2020_20']['ANX_SCORE'] =",county_feats['11001']['2020_20']['ANX_SCORE'])
      available_yws = list(county_feats['11001'].keys())
      available_yws.sort()
      print("\nAvailable weeks for 11001:",  available_yws)

    # Get the closest k_neighbors for each populous_county we want to examine
    county_representation = {}

    matched_counties = {}

    already_matched = []

    for target in populous_counties:

      # Get the k top neighbors
      county_index = list(populous_counties).index(target)
      n_neighbors = neighbors.kneighbors([county_factors[county_index]], k_neighbors + 1, return_distance=False)
      matched_counties[target] = []

      # TODO pick the 1 closest neighbor with a greedy search
      match_found = False
      for i, n in enumerate(n_neighbors[0][1:]): # skip 0th entry (self)

        ith_closest_county = populous_counties[n]

        # determine how much each county appears
        if ith_closest_county not in county_representation.keys():
          county_representation[ith_closest_county] = 0
        county_representation[ith_closest_county] += 1

        if match_found:
          continue

        # TODO filter out counties with county events close by in time
        ith_closest_county_event, _, _ = first_covid_case.get(target,[None,None,None])
        # if abs(ith_closest_county_event - target_event) < event_timing_buffer: continue

        if ith_closest_county not in already_matched:
          matched_counties[target].append(ith_closest_county)
          already_matched.append(ith_closest_county)
          match_found = True

    neighbor_counts = sorted(county_representation.items(), key=lambda kv: kv[1])
    print("\nCount of times each county is a neighbor\n", neighbor_counts[:10],"...",neighbor_counts[-10:], '\n')

    # Calculate Average and Weighted Average Topic Usage
    county_list = county_feats.keys() # all counties considered in order
    county_list_weights = [county_representation.get(c,1) for c in county_list] # weight based on neighbor count
    avg_county_list_usages = np.array([])
    weighted_avg_county_list_usages = np.array([])
    if topics:
      county_list_usages = [] # all averaged counties stacked
      for county in county_list:
        yearweeks = list(county_feats[county].keys())
        county_list_usages.append( avg_topic_usage_from_dates(county,yearweeks) )
      # county_feats[county][year_week] = [feats numpy array]
      avg_county_list_usages = np.average(county_list_usages, axis=0)
      weighted_avg_county_list_usages = np.average(county_list_usages, weights=county_list_weights,  axis=0)
    else:
      pass
      # TODO needs to be implemented
      # county_feats[county][year_week][feat] = value    
    

    # Calculate diff in diffs dict[county] = [feature_array]
    target_befores = {}
    target_diffs = {}
    matched_befores = {}
    matched_diffs = {}
    avg_matched_befores = {}
    avg_matched_diffs = {}
    for target in populous_counties:

      # George Flyod's Death
      #target_event_start, target_event_end, event_name = county_events.get(target,[None,None,None])
      # First Covid Death
      #target_event_start, target_event_end, event_name = first_covid_death.get(target,[None,None,None])
      # First Case of Covid
      target_event_start, target_event_end, event_name = first_covid_case.get(target,[None,None,None])
      

      if target_event_start is None and target_event_end is None:
        continue # no event was found for this county

      target_before, target_after, dates_before, dates_after = feat_usage_before_and_after(target, event_start=target_event_start, event_end=target_event_end)
      if target_before is None or target_after is None: 
        continue # not enough data about this county

      #print('dates_before',dates_before)
      #print('dates_after',dates_after)

      target_diff = np.subtract(target_after,target_before)

      target_befores[target] = target_before
      target_diffs[target] = target_diff

      matched_counties_considered = matched_counties[target]
      matched_diffs[target] = []
      matched_befores[target] = []
      for matched_county in matched_counties_considered:
        matched_before, matched_after, _, _ = feat_usage_before_and_after(matched_county, event_start=target_event_start, event_end=target_event_end)
        if matched_before is None or matched_after is None: continue
        matched_diff = np.subtract(matched_after,matched_before)

        # Add all differences, then divide by num of considered counties
        matched_diffs[target].append(matched_diff)
        matched_befores[target].append(matched_before)
      
      if matched_befores[target] == []:
        continue # this target is not viable (no match data)

      avg_matched_befores[target] = matched_befores[target][0] #np.mean(matched_befores[target], axis=0)

      # print('target_diffs',target_diffs)
      # print('matched_diffs',matched_diffs)

      # Average/std change from all matched counties
      avg_matched_diff = np.mean(matched_diffs[target], axis=0)
      std_matched_diff = np.std(matched_diffs[target], axis=0)
      avg_matched_diffs[target] = avg_matched_diff
      

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
      # print("Intervention Effect:\n", intervention_effects)]

      # Relevant Dates
      begin_before, _ = yearweek_to_dates(min(dates_before))
      _, end_before = yearweek_to_dates(max(dates_before))
      begin_after, _ = yearweek_to_dates(min(dates_after))
      _, end_after = yearweek_to_dates(max(dates_after))
      middle_before = begin_before + (end_before - begin_before)/2
      middle_after = begin_after + (end_after - begin_after)/2

      # Calculate in-between dates and xticks
      x = [middle_before, middle_after]
      xticks = [begin_before, end_before, begin_after, end_after]

      #plot_diff_in_diff_per_county()


    # Aggregate Average Feature Usage


    # Aggregated diff in diff
    all_target_befores = np.stack(target_befores.values(), axis=0)
    all_target_diffs = np.stack(target_diffs.values(), axis=0)
    all_target_afters = all_target_befores + all_target_diffs
    all_avg_matched_befores = np.stack(avg_matched_befores.values(), axis=0)
    all_avg_matched_diffs = np.stack(avg_matched_diffs.values(), axis=0)
    all_avg_matched_afters = all_avg_matched_befores + all_avg_matched_diffs

    # Plot aggregated findings for each feature
    if topics:
      list_features = range(num_feats)
    else:
      list_features = range(num_feats) # TODO fix

    stderr_change_map = {}
    for feature_num in list_features: # run against all features
      plt.clf() # reset plot

      target_before = np.mean(all_target_befores[:,feature_num]) # average of all targets before [scalar]
      target_diff = np.mean(all_target_diffs[:,feature_num]) # average of all target diffs [scalar]
      target_after = target_before + target_diff  # average of all target afters [scalar]

      avg_match_before = np.mean(all_avg_matched_befores[:,feature_num])
      avg_match_diff = np.mean(all_avg_matched_diffs[:,feature_num])
      avg_match_after = avg_match_before + avg_match_diff

      target_expected = target_before + avg_match_diff  # average of all targets_before + avg_matched_diff [scalar]
      intervention_effect = target_after - target_expected
      
      std_match_before = np.std(all_avg_matched_befores[:,feature_num])
      std_match_after = np.std(all_avg_matched_afters[:,feature_num])
      stderr_match_before = std_match_before / np.sqrt(k_neighbors)
      stderr_match_after = std_match_after / np.sqrt(k_neighbors)

      is_significant = abs(intervention_effect) > stderr_match_after*ci_window
      if not is_significant: 
        continue # only plot significant results
      increase_decrease = "increased" if intervention_effect > 0 else "decreased"
      stderr_change = intervention_effect/stderr_match_after
      if topics:
        print("Change in {} {} significantly ({} stderrs) -> Topic #{}".format( \
          topic_map[str(feature_num)][:8], increase_decrease, stderr_change,feature_num)) 
      else:
        print("Change in {} {} significantly ({} stderrs)".format( \
          feature_num, increase_decrease, stderr_change)) 

      stderr_change_map[stderr_change] = feature_num

      # Confidence Intervals
      ci_down = [target_before-stderr_match_before, target_expected-stderr_match_after]
      ci_up = [target_before+stderr_match_before, target_expected+stderr_match_after]
      ci_down_2 = [target_before-stderr_match_before*ci_window, target_expected-stderr_match_after*ci_window]
      ci_up_2 = [target_before+stderr_match_before*ci_window, target_expected+stderr_match_after*ci_window]
      
      # Create Plot
      x = [2, 6]
      xticks = [1, 3, 5, 7]
      fig, ax = plt.subplots()
      fig.set_size_inches(6, 6)
      plt.plot(x, [target_before, target_after], 'b-', label='Target (Actual)')
      plt.plot(x, [target_before, target_expected],'c--', label='Target (Expected)')
      #plt.plot([x[0]]*30, matches_before[:,feature_num], 'r+', alpha=0.2)
      #plt.plot([x[1]]*30, matches_after[:,feature_num], 'r+', alpha=0.2)
      plt.plot(x,[avg_match_before,avg_match_after],'r--',label='Average Match')
      plt.fill_between(x, ci_down, ci_up, color='c', alpha=0.3)
      plt.fill_between(x, ci_down_2, ci_up_2, color='c', alpha=0.2)
      plt.plot([x[1],x[1]], [target_after, target_expected], 'k--', \
        label='Intervention Effect ({})'.format(round(intervention_effect,5)))
      #plt.axhline(y=avg_county_list_usages[feature_num], color='g', linestyle='-',label='Average Topic Usage')
      #plt.axhline(y=weighted_avg_county_list_usages[feature_num], color='g', linestyle='--',label='Weighted Average Topic Usage')
      plt.title("All US Counties Before/After " + event_name)

      # Format plot
      plt.xticks(rotation=45, ha='right')
      ax.set_xticks(xticks)
      ax.set_xticklabels([
        "{} weeks before event".format(default_event_buffer + default_before_start_window + 1),
        "{} week before event".format(default_event_buffer),
        "{} week after event".format(default_event_buffer),
        "{} weeks after event".format(default_event_buffer + default_after_end_window + 1)
      ])
      plt.xlabel("Time".format(dates_before,dates_after))
      if topics:
        plt.ylabel(str(topic_map[str(feature_num)][:4]) + " mentions per 100k")
      else: 
        plt.ylabel(str(feature_num) + " Usage")
      plt.legend()
      plt.tight_layout()
      
      if topics:
        plt_name = "did_topic{}_USA_time_before_after_covid_case.png".format( \
          feature_num)
      else:
        plt_name = "did_{}_USA_time_before_after_covid_case.png".format( \
          feature_num)

      plt.savefig(plt_name)
      if feature_num > 20: break # TODO remove

    # Print out the resuls in sorted order
    print("\nSorted Results:")
    for stderr in sorted(stderr_change_map.keys()):
      feature = stderr_change_map[stderr]
      if topics:
        print("{} was {} stderr away (Topic #{})".format(topic_map[str(feature)],stderr,feature))
      else:
        print("{} was {} stderr away".format(feature, stderr))



      
