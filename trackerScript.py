import os 
from datetime import datetime, timedelta
from time import tzname
import time
from pprint import pprint

import urllib.parse as urlparse
from urllib.parse import urlencode, urlsplit, parse_qs

import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import plotly
import plotly.express as px
import plotly.graph_objs as go

from options import getValidCountyList

random.seed(42)

###################################################
## Aesthetic Options
colorPallete1 = px.colors.qualitative.T10[0:]
colorPallete2 = px.colors.sequential.Blues[1:2]*10

barColors = colorPallete1
lineColors = colorPallete1
lineMarkColors = colorPallete1
lineMarkOutlineColors = colorPallete2


########################################
## Other Settings

countyFixes = { # for aggregate counties as cities
    'New York City': 36061,
    'Kansas City': 29095
    }
seconds_to_store_data = 30
countyDataFilePath = '/data1/covid-interface-prod/covid-interface/assets/covid-19-data/us-counties.csv'



config = {'displaylogo': False}

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

'''
VALID_USERNAME_PASSWORD_PAIRS = {
    'wwbp': 'countycovidtracker',
    'adi': 'countycovidtracker',
    'hlab': 'countycovidtracker',
    'has': 'countycovidtracker'
}
'''

def getLastDataUpdate():
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime('/data1/covid-interface-prod/covid-interface/assets/covid-19-data/us-counties.csv'))) + f" {tzname[0]}"

def getPsswd():
    if('.psswd' in list(os.listdir('/data1/covid-interface-prod/covid-interface/assets/'))):
        print ('Password Exists')
        f = open('/data1/covid-interface-prod/covid-interface/assets/.psswd')
        a = f.readlines()
        passwordDict = {}
        for i in a:
            passwordDict[i.strip().split(':')[0]] = i.strip().split(':')[-1].strip() 
        return passwordDict
    return False

#for caching data
last_update = datetime(2013,12,31,23,59,59)
last_data = None
def getDailyData():
    #filePath = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    global last_update, last_data
    now = datetime.now()
    if (not isinstance(last_data, pd.DataFrame)) or ((now - last_update).total_seconds() < seconds_to_store_data):
        if('.csv' in countyDataFilePath):
            #print ('Reading the file ....')
            data = pd.read_csv(countyDataFilePath, parse_dates=['date'])
            data['state_abbrev'] = data['state'].map(us_state_abbrev)
            #Fixes
            for newName, fips in countyFixes.items():
                data.loc[data.county == newName, 'fips'] = fips
        else:
            data  = None

        #update cache
        last_update = now
        last_data = data

    return last_data
    
def getPopulationData():
    cols = ['fips', 'population', 'county', 'state_abbrev', 'state:county'] 
    populationData = pd.read_csv('/data1/covid-interface-prod/covid-interface/assets/countymap.csv')
    populationData.columns = cols
    #Fixinf population for NYC
    populationData.loc[populationData["fips"]==36061, "population"] = 8537673
    return populationData

def removeCache(dirPath, n=4):
    dirPath = '/data1/covid-interface-prod/covid-interface/assets/' + dirPath
    fileStats = []
    for i in os.listdir(dirPath):
        temp = []
        if (';' in i):
            temp.append(dirPath + '/' + i)
            ctime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(dirPath + '/' + i)))
            temp.append(ctime)
            fileStats.append(temp)
    
    if len(fileStats) < n:
        return 

    print (fileStats)
    fileStats = sorted(fileStats, key=lambda x: x[1])
    filesToDel = fileStats[:n]
    for i in filesToDel:
        try:
            os.remove(i[0])
            print (f'Removed {i[0]}.....')
        except:
            print (f'Couldn\'t remove {i[0]}')
    return


def formatCounty(countyList:list):
    temp = []
    return [i[:-5] for i in countyList]

def getSocialMediaURL(searchString):
    tweetBase = '''https://twitter.com/share'''
    fbBase = '''http://www.facebook.com/sharer.php'''
    urlToShare = '''https://countycovid.com/'''+searchString

    queryDict = getURLQueryDict(searchString)
    if(queryDict['Counties'] == [] or queryDict['Counties'] == [None]):
        return '', ''
    
    fbURLParts = list(urlparse.urlparse(fbBase))
    fbURLParts[4] = urlencode({'u':urlToShare})
    shareURL = urlparse.urlunparse(fbURLParts)
    
    tweetURLParts = list(urlparse.urlparse(tweetBase))
    tweetText = 'COVID19 trends at the'
    for county in queryDict['Counties']:
        tweetText += f' {county.split(":")[-1]} county ({county.split(":")[0]}),'
    tweetText = tweetText[:-1]
    tweetText += "." 
    tweetURLParts[4] = urlencode({'url':urlToShare, 'text':tweetText})
    tweetURL = urlparse.urlunparse(tweetURLParts)
    return shareURL, tweetURL

def getURLQueryString(queryDict:dict):
    url = ''
    url_parts = list(urlparse.urlparse(url))
    query = dict(urlparse.parse_qsl(url_parts[4]))
    url_parts[4] = urlencode(queryDict)    
    queryStr = urlparse.urlunparse(url_parts)
    print (f'query String: {queryStr}')
    return queryStr

def getURLQueryDict(url:str):
    queryDict = parse_qs(urlsplit(url).query)
    for i in queryDict:
        queryDict[i]= eval(queryDict[i][0])
    if 'Counties' in queryDict:
        invalidCounties = set(queryDict['Counties']) - set(queryDict['Counties']).intersection(set(getValidCountyList()))
        if len(invalidCounties)>0:
            validCounties = [county for county in queryDict['Counties'] if county not in invalidCounties]
            validCounties.append(None)
            queryDict['Counties'] = validCounties
    else:
        queryDict['Counties'] = []
    
    return queryDict

def CountyIncreaseCases(countyList, data):
    dates = []
    df = [] 
    for i in countyList:
        tempData = data[data["state:county"]==i].sort_values(['date'], ascending=True)
        tempData['newCases'] = tempData.cases.shift(1).fillna(0)
        tempData['newCases'] = tempData.cases - tempData.newCases
        tempData['newCases'] = tempData.newCases.ewm(alpha=0.3).mean()
        tempData = tempData[tempData.cases>10]
        df.append(tempData)

    data = pd.concat(df, axis=0).reset_index(drop=True)
    data["dates"] = data.date.dt.strftime('%Y-%m-%d')
    dates = sorted(list(data["dates"].unique()))
    
    xRange = [0.8, (np.max(np.log10(data.cases.values)))+0.5]
    yRange = [np.min(np.log10(data.newCases.values)) - 0.2, (np.max(np.log10(data.newCases.values)))+0.2]
    data['newCases'] = data['newCases'].map('{:,.2f}'.format)

    #print (f"xRange: {xRange}")
    #print (f"yRange: {yRange}")
    
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["title"] = "Indicator plot: Are we seeing a drop in the number of new cases?"
    fig_dict["layout"]["template"] = "ggplot2"
    fig_dict["layout"]["plot_bgcolor"] = 'rgb(255,255,255)'
    fig_dict["layout"]["showlegend"] = True
    #fig_dict["layout"]["width"] = 1425
    #fig_dict["layout"]["height"] = 600
    fig_dict["layout"]["xaxis"] = {"title": "Total number of cases - log scale", "type": "log", 'range':xRange, 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}
    fig_dict["layout"]["yaxis"] = {"title": "New cases (Smoothed) - log scale", "type": "log", 'range':yRange, 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["sliders"] = {
        "args": [
            "transition", {
                "duration": 400,
                "easing": "cubic-in-out"
            }
        ],
        "initialValue": dates[0],
        "plotlycommand": "animate",
        "values": dates,
        "visible": True
    }
    fig_dict["layout"]["updatemenus"] = [
        {   "active":0,
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Date:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # make data
    date = dates[1]
    j=0

    for i in countyList:
        dataset_by_date = data[data["dates"] <= date]
        dataset_by_date_and_county = dataset_by_date[
            dataset_by_date["state:county"] == i]

        data_dict = {
            "x": list(dataset_by_date_and_county["cases"]),
            "y": list(dataset_by_date_and_county["newCases"]),
            "mode": "lines",
            'line_shape':'spline',
            'marker':dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=2)), 
            'line':dict(color=lineColors[j], width=4),
            "text": list(dataset_by_date["state:county"]),
            "name": i
        }
        fig_dict["data"].append(data_dict)
        j += 1

    # make frames
    for date in dates:
        frame = {"data": [], "name": date}
        j = 0
        for i in countyList:
            dataset_by_date = data[data["dates"] <= date]
            dataset_by_date_and_county = dataset_by_date[
                dataset_by_date["state:county"] == i]

            data_dict = {
                "x": list(dataset_by_date_and_county["cases"]),
                "y": list(dataset_by_date_and_county["newCases"]),
                "mode": "lines",
                'line_shape':'spline',
                'marker':dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=2)), 
                'line':dict(color=lineColors[j], width=4),
                "text": list(dataset_by_date_and_county["state:county"]),
                "name": i
            }
            frame["data"].append(data_dict)
            j += 1

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [date],
            {"frame": {"duration": 300, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 300}}
        ],
            "label": date,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)


    fig_dict["layout"]["sliders"] = [sliders_dict]

    return fig_dict

def CountyDailyNewDeaths(countyList:list, data, populationData):
    traces = []
    Ys_abs = []
    Ys_100k = []
    Xs = []
    minZoomDate = "2030-12-12"
    captionFlag = 0
    caption = ""
    j=0
    negDates = []
    annotations = []

    for i in countyList:
        
        tempData = data[data['state:county']== i]
        tempData = tempData.sort_values(by=['date'], ascending=True)

        tempData['currentDeaths'] = tempData.deaths.shift(1).fillna(0)
        tempData['currentDeaths'] = tempData.deaths - tempData.currentDeaths
        if tempData[tempData.currentDeaths<0].shape[0]>0:
            negDates.extend(list(tempData[tempData.currentDeaths<0].date.dt.strftime('%m/%d')))
            tempData.loc[tempData.currentDeaths<0,'currentDeaths'] = 0            
            captionFlag=1
        
        currentFips = tempData[tempData['state:county'] == i].fips.values[0]
        currentPopulation = populationData[populationData['fips']==currentFips]['population'].values[0]
        #print (f"County {i} Population: {currentPopulation}")

        X = tempData.date.dt.strftime('%Y-%m-%d')
        Y_abs = tempData.currentDeaths
        Y_100k = (tempData.currentDeaths/currentPopulation*100000).map('{:,.2f}'.format)
        #Y_abs_ma = tempData.currentCases.rolling(window=7).mean()
        #Y_100k_ma = Y_100k.rolling(window=7).mean()
        Y_abs_ma = (tempData.currentDeaths.ewm(alpha=0.33).mean()).map('{:,.2f}'.format)
        Y_100k_ma = (Y_100k.ewm(alpha=0.33).mean()).map('{:,.2f}'.format)
        Xs.append(X)

       
        if(len(countyList)==1):
            trace = [go.Bar(x=X, y=Y_abs, name=i, legendgroup=i, marker_color=barColors[j], opacity=0.66), go.Scatter(x=X, y=Y_abs_ma, name='Smoothed', legendgroup=i, mode='lines+markers', marker=dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=1)), line=dict(color=lineColors[j], width=4))]
        else:
            trace = [go.Bar(x=X, y=Y_100k, name=i, legendgroup=i, marker_color=barColors[j], opacity=0.66), go.Scatter(x=X, y=Y_100k_ma, name='Smoothed', legendgroup=i, mode='lines+markers', marker=dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=1)), line=dict(color=lineColors[j], width=4))]
        j += 1

        if tempData[tempData.deaths > 0].shape[0]>0:
            minZoomDate = min(tempData[tempData.deaths > 0].date.dt.strftime('%Y-%m-%d').values[0], minZoomDate)
        else:
            minZoomDate = min(tempData[tempData.deaths == 0].date.dt.strftime('%Y-%m-%d').values[0], minZoomDate)

        traces.extend(trace)

        Ys_abs.extend([Y_abs, Y_abs_ma])
        Ys_100k.extend([Y_100k, Y_100k_ma])    

    updateMenus = [{'active':1 if len(countyList)>1 else 0,
                    'type':'buttons',
                    'x':0.53,
                    'y':1.18,
                    "xanchor": "center",
                    "yanchor":"top",
                    'direction': 'right',
                    'showactive': True,
                    'pad':{"r": 10, "t": 10, 'b': 10, 'l': 10 },
                    'buttons': [
                                {'method': 'update',
                                'label': 'Daily New Deaths',
                                'args': [
                                        {'y': Ys_abs}, 
                                        {'title':'Daily New Deaths over time', 'yaxis':{'title': 'Number of Deaths', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                {'method': 'update',
                                'label': 'Daily New Deaths per 100k',
                                'args': [
                                        {'y': Ys_100k}, 
                                        {'title':'Daily New Deaths per 100k over time', 'yaxis':{'title': 'Number of Deaths per 100k', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                ],
                    },
                    ]
    
    maxZoomDate = datetime.strptime(X.iloc[-1], '%Y-%m-%d')
    maxZoomDate = maxZoomDate + timedelta(days=1)
    maxZoomDate = datetime.strftime(maxZoomDate, '%Y-%m-%d')

    title = 'Daily New Deaths over time' if(len(countyList)==1) else 'Daily New Deaths per 100k over time'
    y_title  = 'Number of Deaths' if(len(countyList)==1) else 'Number of Deaths per 100k'
    if captionFlag==1:
        #To remove duplicates
        negDates = sorted(list(set(negDates)))
        negDates = ", ".join(negDates) if len(negDates) > 1 else negDates[0]
        caption = f"A decrease in cumulative deaths was observed on {negDates} which is not depicted in the graph above. <br> NYT states this is due to changes in reporting standards for a county on a particular day" 
        annotations = [{'xref':'paper', 'yref':'paper', 'x':0.5, 'y':-0.35, 'showarrow':False, 'text':caption}]
    layout = go.Layout(updatemenus=updateMenus, xaxis={'title':'', 'range':[minZoomDate, maxZoomDate], 'gridcolor':'lightgray','linecolor':'slategray', 'mirror':True}, title=title, yaxis={'title':y_title, 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}, showlegend=True, template='ggplot2', margin={'b':90}, annotations=annotations, font={'size':14}, plot_bgcolor='rgb(255,255,255)')
    fig_or_data = {'data':traces, 'layout':layout}
    return fig_or_data


def CountyDailyNewCases(countyList:list, data, populationData):
    traces = []
    Ys_abs = []
    Ys_100k = []
    Xs = []
    minZoomDate = "2030-12-12"
    captionFlag = 0
    caption = ""
    j=0
    negDates = []
    annotations = []

    for i in countyList:
        
        tempData = data[data['state:county']== i]
        tempData = tempData.sort_values(by=['date'], ascending=True)

        tempData['currentCases'] = tempData.cases.shift(1).fillna(0)
        tempData['currentCases'] = tempData.cases - tempData.currentCases
        if tempData[tempData.currentCases<0].shape[0]>0:
            negDates.extend(list(tempData[tempData.currentCases<0].date.dt.strftime('%m/%d')))
            tempData.loc[tempData.currentCases<0,'currentCases'] = 0            
            captionFlag=1
        
        currentFips = tempData[tempData['state:county'] == i].fips.values[0]
        currentPopulation = populationData[populationData['fips']==currentFips]['population'].values[0]
        #print (f"County {i} Population: {currentPopulation}")

        X = tempData.date.dt.strftime('%Y-%m-%d')
        Y_abs = tempData.currentCases
        Y_100k = (tempData.currentCases/currentPopulation*100000).map('{:,.2f}'.format)
        #Y_abs_ma = tempData.currentCases.rolling(window=7).mean()
        #Y_100k_ma = Y_100k.rolling(window=7).mean()
        Y_abs_ma = (tempData.currentCases.ewm(alpha=0.33).mean()).map('{:,.2f}'.format)
        Y_100k_ma = (Y_100k.ewm(alpha=0.33).mean()).map('{:,.2f}'.format)
        Xs.append(X)

        if(len(countyList)==1):
            trace = [go.Bar(x=X, y=Y_abs, name=i, legendgroup=i, marker_color=barColors[j], opacity=0.66), go.Scatter(x=X, y=Y_abs_ma, name='Smoothed', legendgroup=i, mode='lines+markers', marker=dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=1)), line=dict(color=lineColors[j], width=4))]
        else:
            trace = [go.Bar(x=X, y=Y_100k, name=i, legendgroup=i, marker_color=barColors[j], opacity=0.66), go.Scatter(x=X, y=Y_100k_ma, name='Smoothed', legendgroup=i, mode='lines+markers', marker=dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=1)), line=dict(color=lineColors[j], width=4))]
        j += 1

        minZoomDate = min(tempData[tempData.cases <= 10].date.dt.strftime('%Y-%m-%d').values[-1], minZoomDate)

        traces.extend(trace)

        Ys_abs.extend([Y_abs, Y_abs_ma])
        Ys_100k.extend([Y_100k, Y_100k_ma])    

    updateMenus = [{'active':1 if len(countyList)>1 else 0,
                    'type':'buttons',
                    'x':0.53,
                    'y':1.18,
                    "xanchor": "center",
                    "yanchor":"top",
                    'direction': 'right',
                    'showactive': True,
                    'pad':{"r": 10, "t": 10, 'b': 10, 'l': 10 },
                    'buttons': [
                                {'method': 'update',
                                'label': 'Daily New Cases',
                                'args': [
                                        {'y': Ys_abs}, 
                                        {'title':'Daily New Cases over time', 'yaxis':{'title': 'Number of Cases', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                {'method': 'update',
                                'label': 'Daily New Cases per 100k',
                                'args': [
                                        {'y': Ys_100k}, 
                                        {'title':'Daily New Cases per 100k over time', 'yaxis':{'title': 'Number of Cases per 100k', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                ],
                    },
                    ]
    
    maxZoomDate = datetime.strptime(X.iloc[-1], '%Y-%m-%d')
    maxZoomDate = maxZoomDate + timedelta(days=1)
    maxZoomDate = datetime.strftime(maxZoomDate, '%Y-%m-%d')

    title = 'Daily New Cases over time' if(len(countyList)==1) else 'Daily New Cases per 100k over time'
    y_title  = 'Number of Cases' if(len(countyList)==1) else 'Number of Cases per 100k'
    if captionFlag==1:
        #To remove duplicates
        negDates = sorted(list(set(negDates)))
        negDates = ", ".join(negDates) if len(negDates) > 1 else negDates[0]
        caption = f"A decrease in cumulative cases was observed on {negDates} which is not depicted in the graph above. <br> NYT states this is due to changes in reporting standards for a county on a particular day" 
        annotations = [{'xref':'paper', 'yref':'paper', 'x':0.5, 'y':-0.35, 'showarrow':False, 'text':caption}]
    layout = go.Layout(updatemenus=updateMenus, xaxis={'title':'', 'range':[minZoomDate, maxZoomDate], 'gridcolor':'lightgray','linecolor':'slategray', 'mirror':True}, title=title, yaxis={'title':y_title, 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}, showlegend=True, template='ggplot2', margin={'b':90}, annotations=annotations, font={'size':14}, plot_bgcolor='rgb(255,255,255)')
    fig_or_data = {'data':traces, 'layout':layout}
    return fig_or_data

def CountyCumulativeDeaths(countyList:list, data, populationData):    
    traces = []
    Ys_abs = []
    Ys_100k = []
    Ys_abs_log = []
    Xs_log = []
    Xs = []
    j=0
    minZoomDate = "2030-12-12"

    for i in countyList:        
        tempData = data[data['state:county']== i]
        tempData = tempData.sort_values(by=['date'], ascending=True)
 
        currentFips = tempData[tempData['state:county'] == i].fips.values[0]
        currentPopulation = populationData[populationData['fips']==currentFips]['population'].values[0]
        #print (f"County {i} Population: {currentPopulation}")
        #     barColors = colorPallete1
# lineColors = colorPallete1
# lineMarkColors = colorPallete1
# lineMarkOutlineColors = colorPallete1


        
        X = tempData[tempData.deaths > 0].date.dt.strftime('%Y-%m-%d') if(tempData[tempData.deaths > 0].shape[0]>0) else tempData.date.dt.strftime('%Y-%m-%d')
        Y_abs = tempData[tempData.deaths > 0].deaths.values if(tempData[tempData.deaths > 0].shape[0]>0) else tempData.deaths.values
        Y_100k = (tempData[tempData.deaths > 0].deaths/currentPopulation*100000).map('{:,.2f}'.format) if(tempData[tempData.deaths > 0].shape[0]>0) else (tempData.deaths/currentPopulation*100000).map('{:,.2f}'.format) 
        trace = go.Scatter(x=X, y=Y_abs, name=i, mode='lines+markers', marker=dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=1)), line=dict(color=lineColors[j], width=4))  if len(countyList) == 1 else go.Scatter(x=X, y=Y_100k, name=i, mode='lines+markers', marker=dict(color=lineMarkColors[j], line=dict(color=lineMarkColors[j],width=1)), line=dict(color=lineColors[j], width=4))
        j+=1
        if tempData[tempData.deaths > 0].shape[0]>0:
            minZoomDate = min(tempData[tempData.deaths > 0].date.dt.strftime('%Y-%m-%d').values[0], minZoomDate)
        else:
            minZoomDate = min(tempData[tempData.deaths == 0].date.dt.strftime('%Y-%m-%d').values[0], minZoomDate)
        #minZoomDate = max(tempData[tempData.deaths <= 2].date.dt.strftime('%Y-%m-%d').values[-1], minZoomDate)
        
        traces.append(trace)

        Xs.append(X)
        Ys_abs.append(Y_abs)
        Ys_100k.append(Y_100k)    

        Xs_log.append(tempData[tempData.deaths > 0].date.dt.strftime(('%Y-%m-%d')))
        Y_abs_log = np.around(np.log10(tempData[tempData.deaths > 0].cases.values), decimals=2)
        Ys_abs_log.append(Y_abs_log)
    
    
    updateMenus = [{'active':1 if len(countyList)>1 else 0,
                    'type':'buttons',
                    'x':0.53,
                    'y':1.18,
                    "xanchor": "center",
                    "yanchor":"top",
                    'direction': 'right',
                    'showactive': True,
                    'pad':{"r": 10, "t": 10, 'b': 10, 'l': 10 },
                    'buttons': [
                                {'method': 'update',
                                'label': 'Cumulative Number of Deaths',
                                'args': [
                                        {'y': Ys_abs, 'x': Xs}, 
                                        {'title':'Cumulative Deaths over time', 'yaxis':{'title': 'Number of Deaths', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                {'method': 'update',
                                'label': 'Cumulative Number of Deaths per 100k',
                                'args': [
                                    {'y': Ys_100k, 'x':Xs}, 
                                    {'title':'Cumulative Deaths per 100k over time', 'yaxis':{'title': 'Number of Deaths per 100k', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                {'method': 'update',
                                'label': 'Cumulative Number of Deaths : log scale',
                                'args': [
                                    {'y': Ys_abs_log, 'x': Xs_log}, 
                                    {'title':'Cumulative Deaths in log scale over time', 'yaxis':{'title': 'log Number of Deaths', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                ],
                    },
                    ]
    maxZoomDate = datetime.strptime(X.iloc[-1], '%Y-%m-%d')
    maxZoomDate = maxZoomDate + timedelta(days=1)
    maxZoomDate = datetime.strftime(maxZoomDate, '%Y-%m-%d')

    title = 'Cumulative Deaths over time' if(len(countyList)==1) else 'Cumulative Deaths per 100k over time'
    y_title  = 'Number of Deaths' if(len(countyList)==1) else 'Number of Deaths per 100k'

    layout = go.Layout(updatemenus=updateMenus, xaxis={'title':'', 'range':[minZoomDate, maxZoomDate], 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}, title=title, yaxis={'title':y_title, 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}, showlegend=True, template='ggplot2',  font={'size':14}, plot_bgcolor='rgb(255,255,255)')
    fig_or_data = {'data':traces, 'layout':layout}
    
    return fig_or_data


def CountyCumulativeCases(countyList:list, data, populationData):    
    traces = []
    Ys_abs = []
    Ys_100k = []
    Ys_abs_log = []
    Xs_log = []
    Xs = []
    j=0
    minZoomDate = "2030-12-12"

    for i in countyList:        
        tempData = data[data['state:county']== i]
        tempData = tempData.sort_values(by=['date'], ascending=True)
 
        currentFips = tempData[tempData['state:county'] == i].fips.values[0]
        currentPopulation = populationData[populationData['fips']==currentFips]['population'].values[0]
        #print (f"County {i} Population: {currentPopulation}")

        X = tempData[tempData.cases > 0].date.dt.strftime('%Y-%m-%d')
        Y_abs = tempData[tempData.cases > 0].cases.values 
        Y_100k = (tempData[tempData.cases > 0].cases/currentPopulation*100000).map('{:,.2f}'.format) 
        trace = go.Scatter(x=X, y=Y_abs, name=i, mode='lines+markers', marker=dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=1)), line=dict(color=lineColors[j], width=4))  if len(countyList) == 1 else go.Scatter(x=X, y=Y_100k, name=i, mode='lines+markers', marker=dict(color=lineMarkColors[j], line=dict(color=lineMarkOutlineColors[j],width=1)), line=dict(color=lineColors[j], width=4))
        j+=1
        minZoomDate = min(tempData[tempData.cases <= 10].date.dt.strftime('%Y-%m-%d').values[-1], minZoomDate)
        
        traces.append(trace)

        Xs.append(X)
        Ys_abs.append(Y_abs)
        Ys_100k.append(Y_100k)    

        Xs_log.append(tempData[tempData.cases > 10].date.dt.strftime(('%Y-%m-%d')))
        Y_abs_log = np.around(np.log10(tempData[tempData.cases > 10].cases.values), decimals=2)
        Ys_abs_log.append(Y_abs_log)
    
    
    updateMenus = [{'active':1 if len(countyList)>1 else 0,
                    'type':'buttons',
                    'x':0.53,
                    'y':1.18,
                    "xanchor": "center",
                    "yanchor":"top",
                    'direction': 'right',
                    'showactive': True,
                    'pad':{"r": 10, "t": 10, 'b': 10, 'l': 10 },
                    'buttons': [
                                {'method': 'update',
                                'label': 'Cumulative Number of Cases',
                                'args': [
                                        {'y': Ys_abs, 'x': Xs}, 
                                        {'title':'Cumulative Cases over time', 'yaxis':{'title': 'Number of Cases', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                {'method': 'update',
                                'label': 'Cumulative Number of Cases per 100k',
                                'args': [
                                    {'y': Ys_100k, 'x':Xs}, 
                                    {'title':'Cumulative Cases per 100k over time', 'yaxis':{'title': 'Number of Cases per 100k', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                {'method': 'update',
                                'label': 'Cumulative Number of Cases : log scale',
                                'args': [
                                    {'y': Ys_abs_log, 'x': Xs_log}, 
                                    {'title':'Cumulative Cases in log scale over time', 'yaxis':{'title': 'log Number of Cases', 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}}
                                        ]
                                },
                                ],
                    },
                    ]
    maxZoomDate = datetime.strptime(X.iloc[-1], '%Y-%m-%d')
    maxZoomDate = maxZoomDate + timedelta(days=1)
    maxZoomDate = datetime.strftime(maxZoomDate, '%Y-%m-%d')

    title = 'Cumulative Cases over time' if(len(countyList)==1) else 'Cumulative Cases per 100k over time'
    y_title  = 'Number of Cases' if(len(countyList)==1) else 'Number of Cases per 100k'

    layout = go.Layout(updatemenus=updateMenus, xaxis={'title':'', 'range':[minZoomDate, maxZoomDate], 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}, title=title, yaxis={'title':y_title, 'gridcolor':'lightgray', 'linecolor':'slategray', 'mirror':True}, showlegend=True, template='ggplot2',  font={'size':14}, plot_bgcolor='rgb(255,255,255)')
    fig_or_data = {'data':traces, 'layout':layout}
    
    return fig_or_data


def getGraphs(countyList:list, baseOptSelected:str = 'New Cases - County level'):
    graphType = 1 if baseOptSelected == 'County Cases' else 2

    #Remove this later
    #countyList = formatCounty(countyList)
    #print (f'Counties: {countyList}')
    data = getDailyData() #reads the csv for data
    data['state:county'] = data['state_abbrev']+':'+data['county']
    data = data[data['state:county'].isin(list(countyList))]
    data = data.sort_values(['date'], ascending=True)
    
    populationData = getPopulationData()

    if (baseOptSelected == 'Total Cases - County level'):
        fig_or_data = CountyCumulativeCases(countyList, data, populationData)
    elif(baseOptSelected == 'New Cases - County level'):
        fig_or_data = CountyDailyNewCases(countyList, data, populationData)
    elif(baseOptSelected == 'Increase Trend'):
        fig_or_data = CountyIncreaseCases(countyList, data)
    elif(baseOptSelected == 'Total Deaths - County level'):
        fig_or_data = CountyCumulativeDeaths(countyList, data, populationData)
    elif(baseOptSelected == 'New Deaths - County level'):
        fig_or_data = CountyDailyNewDeaths(countyList, data, populationData)

    graphJS =  str(plotly.offline.plot(fig_or_data, image_width='100%', image_height='70%', include_plotlyjs=False, output_type='div', auto_open=False))
    graphJS = '''<html> \n <head> <style> body {background-color: lightgray;} </style> \n <script src="https://cdn.plot.ly/plotly-latest.min.js"></script> \n</head>\n<body>'''+ graphJS.replace("'", '"') +'''\n</body></html>'''


    return go.Figure(fig_or_data)
    '''
    traces_absCases = []
    traces_100kCases = []
    Ys_absCases = []
    Ys_100kCases = []

    for i in countyList:
        
        tempData = data[data['state:county']== i]
        tempData = tempData.sort_values(by=['date'], ascending=True)

        tempData['currentCases'] = tempData.cases.shift(1).fillna(0)
        tempData['currentCases'] = tempData.cases - tempData.currentCases
    
        tempData['currentDeaths'] = tempData.deaths.shift(1).fillna(0)
        tempData['currentDeaths'] = tempData.deaths - tempData.currentDeaths
        
        currentFips = tempData[tempData['state:county'] == i].fips.values[0]
        currentPopulation = populationData[populationData['fips']==currentFips]['population'].values[0]
        print (f"County {i} Population: {currentPopulation}")

        X = tempData.date.dt.strftime('%Y-%m-%d').values
        Y_absCases = tempData.cases if graphType == 1 else tempData.currentCases
        Y_100kCases = tempData.cases.values/currentPopulation*100000 if graphType == 1 else tempData.currentCases.values/currentPopulation*100000

        
        trace_absCases = go.Scatter(x=X, y=Y_absCases, name=i) if graphType == 1 else go.Bar(x=X, y=Y_absCases, name=i)

        traces_absCases.append(trace_absCases)

        Ys_absCases.append(Y_absCases)
        Ys_100kCases.append(Y_100kCases)

    label = [['Cumulative Number of Cases', 'Cumulative Number of Cases per 100k'],
            ['Daily New Cases', 'Daily New Cases per 100k']]

    title = [['Cumulative Cases over time', 'Cumulative Cases per 100k over time'], 
            ['Daily New Cases over time', 'Daily New Cases per 100k over time']]
    


    updateMenus = [{'active':0,
                    'type':'buttons',
                    'x':0.7,
                    'y':1.2,
                    'direction': 'right',
                    'showactive': True,
                    'buttons': [{'method': 'update',
                                'label': label[graphType-1][0],
                                'args': [
                                        {'y': Ys_absCases}, 
                                        {'title':title[graphType-1][0], 'yaxis':{'title': 'Number of Cases'}}
                                        ]
                                },
                                {'method': 'update',
                                'label': label[graphType-1][1],
                                'args': [
                                    {'y': Ys_100kCases}, 
                                    {'title':title[graphType-1][1], 'yaxis':{'title': 'Number of Cases per 100k'}}
                                        ]
                                },
                                ],
                    'pad':{"r": 10, "t": 10, 'b': 10, 'l': 10 },
    
                    }]
    layout = go.Layout(updatemenus=updateMenus, xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Date")), title=title[graphType-1][0], yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Cases")))
    fig_or_data = {'data':traces_absCases, 'layout':layout}
    graphJS =  str(plotly.offline.plot(fig_or_data, image_width='100%', image_height='70%', include_plotlyjs=False, output_type='div', auto_open=False))
    #adding html tags removed
    

    fileName = ''
    if len(countyList)>1:
            for i in countyList:
                fileName = fileName + i.split(':')[0] + ':' + i.split(':')[-1][0] + str(len(i.split(':')[-1])) + ';'
    else:
        fileName = countyList[0]+';'

    fileName = fileName[:-1] + '.html'
    folderName = 'County Cases/' if graphType == 1 else 'County Daily New Cases/' 
    fileName = folderName + fileName

    print (f'fileName: {fileName}')
    removeCache(fileName.split('/')[0])
    with open(f'/data1/covid-interface-prod/covid-interface/assets/{fileName}', 'w') as f:
        f.write(graphJS)
    f.close()

    print(f'Graphs Stored to ./assets/{fileName}')
    return fileName
    '''
