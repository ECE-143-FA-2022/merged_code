import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid",{'axes.axisbelow': False,'grid.color': 'w','axes.spines.bottom': False, 'axes.spines.left': False, 'axes.spines.right': False,
 'axes.spines.top': False})
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, widgets, Layout
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Image

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plotly.graph_objects as go

from thefuzz import fuzz
from IPython.display import Image
from ipywidgets import interactive
from collections import defaultdict
import pandas as pd
import os
import sys

import itertools
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interactive, widgets, Layout
from mpl_chord_diagram import chord_diagram

import pandas as pd
import numpy as np
import seaborn as sns
#sns.set_style("whitegrid",{'axes.axisbelow': False,'grid.color': 'w','axes.spines.bottom': False, 'axes.spines.left': False, 'axes.spines.right': False,
# 'axes.spines.top': False})
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, widgets, Layout, fixed
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import pandas_alive


def plotLocations(data2):
    """
    Plots the locations of military interstate disputes
    :param data2: COW interstate disputes dataset
    :type data2: Pandas Dataframe
    """
    
    fig = px.scatter_geo(data2,
                         lon = 'longitude',
                         lat = 'latitude',
                         color='precision',
                         hover_name='countries',
                         animation_frame="year",
                         color_continuous_scale = px.colors.sequential.Bluered,
                         basemap_visible = True,
                         title = "Total number of disputes started in each year")

    fig.update_geos(
    projection_type="mercator",
    showcountries=True, 
    countrycolor="Black",
    lataxis_range=[-30,86],
    )
    fig.update_layout(
    width=900,
    height=700, 
    margin={"r":0,"t":0,"l":0,"b":0},
    showlegend=False,
    )
    return fig

def barTotalDisputesUpto(data,year):

    #plotting countries with most involvements in disputes
    data_for_years = data[data['year']<=year]
    disputes_namea = pd.DataFrame(data_for_years['namea'].value_counts())
    top_10_countries = disputes_namea.head(10)
    xaxis = list(top_10_countries.index)
    yaxis = top_10_countries['namea']
    fig = sns.barplot(x = xaxis, y = yaxis, palette = 'Spectral')
    plt.title('Countries involved in most disputes from 1816 to ' + str(year))
    plt.xlabel('Countries')
    plt.ylabel('Number of Disputes')
    plt.show(fig)

def is_float(x: str):
    """
    Determine if the string `x` can be converted to float
    """
    try:
        float(x)
        return True
    except ValueError:
        return False

def has_char(x: str):
    """
    Check if string `x` contains any alphabetical characters
    """
    m = re.search('[a-zA-Z]', x)
    return m is not None

def str_best_match(s: str, strings):
    """
    Find the best match of string `s` in the list of strings.
    """
    assert isinstance(s, str) and isinstance(strings, list)
    s = s.lower()
    strings = [ss.lower() for ss in strings]
    candidates = [ss for ss in strings if ss.find(s) >= 0 or s.find(ss) >= 0]
    if len(candidates) == 0:
        return None
    match_score = [(fuzz.ratio(s, ss), ss) for ss in candidates]
    best_match = sorted(match_score, key=lambda x: x[0])[-1]
    idx = strings.index(best_match[1])
    return idx


class AllianceMap(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fig = go.Figure()
        # load country attributes such as longitude, latitude, and BGN names
        self.countries_data = pd.read_csv("./version4.1_csv/cow.txt", sep=";", skiprows=28)
        self.countries_data.BGN_proper = self.countries_data.BGN_proper.apply(lambda x: x.strip(" "))
        self.countries_data.BGN_name = self.countries_data.BGN_name.apply(lambda x: x.strip(" "))
        self.decide_countries_on_map()
        self.BGN_name = self.countries_data.BGN_name.tolist()
        self.BGN_proper = self.countries_data.BGN_proper.tolist()
        self.countries_name = self.BGN_name + self.BGN_proper

    def decide_countries_on_map(self):
        """
        Decide which countries can be labelled on the map. 
        Only label those with a land area larger than `min_size`.
        """
        self.countries_on_map = set()
        data = self.countries_data
        idx_to_use = []

        for ix, country in data.iterrows():
            if has_char(country.BGN_proper.lower()):
                self.countries_on_map.add(country.BGN_proper.lower())
                self.countries_on_map.add(country.BGN_name.lower())
                idx_to_use.append(ix)
                # print(country.BGN_proper.lower(), pos)
        self.countries_on_map = list(self.countries_on_map)
        self.countries_data = self.countries_data.iloc[idx_to_use]

    def best_match_country(self, country: str):
        """
        Find the best match of `country` in self.countries_name
        """
        match_idx = str_best_match(country.lower(), self.countries_name)
        if match_idx is None:
            return
        name = self.countries_name[match_idx]
        if match_idx >= len(self.BGN_name):
            match_idx -= len(self.BGN_name)
        return match_idx, name

    def label_countries(self):
        """
        Label `country` with their name on the map. Only print 
        the first `max_len` char of the country names.
        """
        country_label_obj = self.scatter_points(self.countries_data, 'longitude', 'latitude', 'BGN_proper', 'country/region')
        self.fig.add_trace(country_label_obj)

    @staticmethod
    def scatter_points(df, lon_key, lat_key, text_key, name=None):
        graph_object = go.Scattergeo(
            lon = df[lon_key],
            lat = df[lat_key],
            hoverinfo = 'text',
            text = df[text_key],
            mode = 'markers',
            marker = dict(
                size = 2,
                color = 'rgb(255, 0, 0)',
                line = dict(
                    width = 3,
                    color = 'rgba(68, 68, 68, 0)'
                )
            ),
            name = name
        )
        return graph_object

    def get_country_row(self, country):
        """
        Query the attributes of `country`
        """
        ct = country.lower()
        match_i = str_best_match(ct, self.countries_on_map)
        if match_i is None:
            return None
        match_idx, name = self.best_match_country(ct)
        if name.lower() != self.countries_on_map[match_i]:
            return
        row = self.countries_data.iloc[match_idx]
        # print(ct, row)
        return row

    def get_longitude(self, country):
        r0 = self.get_country_row(country)
        if r0 is None:
            return
        coord0 = r0.longitude.item(), r0.latitude.item()
        return r0.longitude.item()

    def get_latitude(self, country):
        r0 = self.get_country_row(country)
        if r0 is None:
            return
        return r0.latitude.item()

    def connect_countries(self, year, key="dyad_st_year"):
        # load the alliances data
        alliance = pd.read_csv("./version4.1_csv/alliance_v4.1_by_dyad.csv")
        year_data = alliance.loc[alliance[key] == year]
        df = year_data.copy()
        df['start_lon'] = df['state_name1'].apply(self.get_longitude)
        df['end_lon'] = df['state_name2'].apply(self.get_longitude)
        df['start_lat'] = df['state_name1'].apply(self.get_latitude)
        df['end_lat'] = df['state_name2'].apply(self.get_latitude)
        df = df.loc[df['start_lon'].notnull() & df['end_lon'].notnull()]
        self._connect(df.loc[df['defense'] == 1], 'rgb(255,0,0)', name='defense')
        self._connect(df.loc[(df['defense'] == 0) & (df['neutrality'] == 1)], 'rgb(0, 255, 0)', name='neutrality')
        self._connect(
            df.loc[(df['defense'] == 0) & (df['neutrality'] == 0) & (df['nonaggression'] == 1)], 
            'rgb(0, 0, 255)', 
            name='nonaggression'
        )
        self._connect(
            df.loc[(df['defense'] == 0) & (df['neutrality'] == 0) & (df['nonaggression'] == 0) & df['entente'] == 1], 
            'rgb(255,165,0)', 
            name='entente'
        )
        return len(df)

    def _connect(self, df, color='#0099C6', name=None):
        lons = np.empty(3 * len(df))
        lons[::3] = df['start_lon']
        lons[1::3] = df['end_lon']
        lons[2::3] = None
        lats = np.empty(3 * len(df))
        lats[::3] = df['start_lat']
        lats[1::3] = df['end_lat']
        lats[2::3] = None
        if len(df) == 0:
            lons = np.array([None])
            lats = np.array([None])

        self.fig.add_trace(
            go.Scattergeo(
                lon = lons,
                lat = lats,
                mode = 'lines',
                hoverinfo = 'none',
                line = dict(width=1.2,color=color),
                opacity = 0.9,
                name = name,
            )
        )

def save_to_img(year):
    """
    Plot the alliance map of `year` and save it to png.
    """
    alliance_map = AllianceMap()
    num = alliance_map.connect_countries(year)
    if num == 0:
        return
    alliance_map.label_countries()
    alliance_map.fig.update_geos(
        projection_type="mercator",
        showcountries=True, 
        countrycolor="Black",
        landcolor="rgb(243,243,243)",
        lataxis_range=[-30,86],
    )
    alliance_map.fig.update_layout(
        title=str(year),
        width=900,
        height=600,
        margin={"r":0,"t":30,"l":0,"b":0},
#         showlegend=False,
    )
    alliance_map.fig.write_image(f"./plotly2/{year}.png", scale=1.5)

def sankey_new(data, name_to_code, year,coun_list):
    """
    Produces graph model in the form of a sankey diagram
    nodes exist on the outer perimiter of a circle
    edges are inside of said circle
    """
   
    list1=list(coun_list)
    coun_list=[name_to_code[n]for n in list1]
    
    countries=defaultdict(int)
    coun_name=[]
    for i in range(len(coun_list)):
        for idx in range(len(data)):
        
            if data[idx][1] ==coun_list[i]:
                coun_name.append(data[idx][0])
                break


    data1 = pd.read_csv('Diplomatic_Exchange_2006v1.csv', sep=",")
    data1 = data1.values.tolist()

    exchange_in_year=defaultdict(int)
    for idx in range(len(data1)):
        if (int(data1[idx][0]),int(data1[idx][1]))in itertools.permutations(coun_list,2):
            if int(data1[idx][2])>=int(year) and int(data1[idx][2])<int(year)+10:
                exchange_in_year[(data1[idx][0],data1[idx][1])]+=data1[idx][4]


    exchange=np.zeros((len(coun_list),len(coun_list)))
    for i in range(len(coun_list)):
        
        for j in range(len(coun_list)):
            
            exchange[i][j]=exchange_in_year[(coun_list[i],coun_list[j])]
    Num=len(coun_list)
    N1=Num
    f=10-int(Num/10)
    chord_diagram(exchange, coun_name,fontsize=f)    

def mapImportExportByYear(data, year, category, normalization):
    """
    Plots a spatio-temporal world map where each country has a certain density based on their imports and exports
    """
    # Import Geopandas World Dataset to map countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # testing out a different projection, this one will drop Antarctica
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands")]

    world = world.to_crs("EPSG:3395")
    fig, (ax,ax2) = plt.subplots(2,1, figsize=(12,12), gridspec_kw = {'height_ratios': [4, 1],'width_ratios':[1]})
    year_subset = data[data["year"]==year]
    vmin = None
    vmax = None
    
    if normalization and (category == "imports" or category == "exports"):
        year_subset[category] = year_subset[category]/year_subset[category].sum()
        vmin = 0.0
        vmax = 0.30
    elif normalization and category == "imports_yearly_difference":
        category = "imports_yearly_percentage_difference"
        vmin = -1.0
        vmax = 1.0
        
    elif normalization and category == "exports_yearly_difference":
        category = "exports_yearly_percentage_difference"
        vmin = -1.0
        vmax = 1.0
        
        
    elif category == "net(exports-imports)" and not normalization:
        # forcing the center to be zero
        vmax = min(year_subset[category].max(),abs(year_subset[category].min()))
        vmin = vmax * -1
    # this will create a biased normalization since the net pos and net negatives will be different
    # but in both cases I am normalizing them down to [-1,1]
    elif category == "net(exports-imports)" and normalization:
        year_subset[category] = year_subset[category]/(year_subset["imports"]+year_subset["exports"])
        vmin = -1
        vmax = 1
            
    
    data_pivot_total = pd.pivot_table(year_subset, index = "country", values = category,
                                     aggfunc = "sum", fill_value = 0)
    
    world_subplot = world.merge(data_pivot_total, left_on="name",right_on = "country", how="left").fillna(value=0)
    print(category)
    world_subplot.plot(ax=ax,legend=True,vmin = vmin,vmax=vmax,column=category,cmap="coolwarm",edgecolor = 'black')
    
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_title("Year: " + str(year), fontsize = 20, loc = "left")
    
    ax2.set_xlim(data["year"].min(),data["year"].max())
    ax2.set_ylim(0,10)
    y = 5
    plt.hlines(y,data["year"].min(),data["year"].max())
    plt.vlines(data["year"].min(), y - 4 / 2., y + 4 / 2.)
    plt.vlines(data["year"].max(), y - 4 / 2., y + 4 / 2.)
    plt.plot(year,y,'ro',ms=12, mfc='r')
    plt.text(data["year"].min() - 0.4, y, str(data["year"].min()), horizontalalignment='right')
    plt.text(data["year"].max() + 0.4, y, str(data["year"].max()), horizontalalignment='left')
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    plt.annotate(str(year),(year,y), xytext = (year,y-3))
    plt.annotate('Year:',(data["year"].min(),y), xytext = (data["year"].min()-6,y-3))
    
    
    
    #print("Top 5 countries")
    
    top5 = data_pivot_total.sort_values(by = category , ascending = False)[:5]
    i = 1
    print("Highest 5 - ")
    for c, v in zip(top5.index, top5[category].values):
        
        test = ''
        test = str(round(v,3))
        if normalization:
            test = str(round(v*100.0,3))+'%'
        print(str(i) + ". " + str(c) + " - " + test)
        i += 1
    if category == "net(exports-imports)" and normalization:
        print("\n\n Lowest 5 - ")
        top5 = data_pivot_total.sort_values(by = category , ascending = True)[:5]

        i = 1
        for c, v in zip(top5.index, top5[category].values):
            test = ''
            test = str(round(v,3))
            if normalization:
                test = str(round(v*100.0,3))+'%'
            print(str(i) + ". " + str(c) + " - " + test)
            i += 1
    
    plt.show()

def mapImportExportByYearAnimation(data,year, category, normalization):
    # Import Geopandas World Dataset to map countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # testing out a different projection, this one will drop Antarctica
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands")]

    world = world.to_crs("EPSG:3395")
    fig, (ax,ax2) = plt.subplots(2,1, figsize=(12,12), gridspec_kw = {'height_ratios': [4, 1],'width_ratios':[1]})
    year_subset = data[data["year"]==year]
    vmin = None
    vmax = None
    
    if normalization and (category == "imports" or category == "exports"):
        year_subset[category] = year_subset[category]/year_subset[category].sum()
        vmin = 0.0
        vmax = 0.30
    elif normalization and category == "imports_yearly_difference":
        category = "imports_yearly_percentage_difference"
        vmin = -1.0
        vmax = 1.0
        
    elif normalization and category == "exports_yearly_difference":
        category = "exports_yearly_percentage_difference"
        vmin = -1.0
        vmax = 1.0
        
        
    elif category == "net(exports-imports)" and not normalization:
        # forcing the center to be zero
        vmax = min(year_subset[category].max(),abs(year_subset[category].min()))
        vmin = vmax * -1
    # this will create a biased normalization since the net pos and net negatives will be different
    # but in both cases I am normalizing them down to [-1,1]
    elif category == "net(exports-imports)" and normalization:
        year_subset[category] = year_subset[category]/(year_subset["imports"]+year_subset["exports"])
        vmin = -1
        vmax = 1
            
    
    data_pivot_total = pd.pivot_table(year_subset, index = "country", values = category,
                                     aggfunc = "sum", fill_value = 0)
    
    world_subplot = world.merge(data_pivot_total, left_on="name",right_on = "country", how="left").fillna(value=0)
    print(category)
    world_subplot.plot(ax=ax,legend=True,vmin = vmin,vmax=vmax,column=category,cmap="coolwarm",edgecolor = 'black')
    
    ax.set_xticks([])
    ax.set_yticks([])
    normalize_str = ''
    if normalization:
        normalize_str = " Normalized"
    ax.set_title("Category: " + category + normalize_str, fontsize = 20, loc = "left")
    
    ax2.set_xlim(data["year"].min(),data["year"].max())
    ax2.set_ylim(0,10)
    y = 5
    plt.hlines(y,data["year"].min(),data["year"].max())
    plt.vlines(data["year"].min(), y - 4 / 2., y + 4 / 2.)
    plt.vlines(data["year"].max(), y - 4 / 2., y + 4 / 2.)
    plt.plot(year,y,'ro', mfc='r')
    plt.text(data["year"].min() - 0.4, y, str(data["year"].min()), horizontalalignment='right')
    plt.text(data["year"].max() + 0.4, y, str(data["year"].max()), horizontalalignment='left')
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    plt.annotate(str(year),(year,y), xytext = (year,y-4))
    plt.annotate('Year:',(data["year"].min(),y), xytext = (data["year"].min()-9,y-4))
    
    
    
    #print("Top 5 countries")
    
    top5 = data_pivot_total.sort_values(by = category , ascending = False)[:5]
    i = 1
    print("Highest 5 - ")
    for c, v in zip(top5.index, top5[category].values):
        
        test = ''
        test = str(round(v,3))
        if normalization:
            test = str(round(v*100.0,3))+'%'
        print(str(i) + ". " + str(c) + " - " + test)
        i += 1
    if category == "net(exports-imports)" and normalization:
        print("\n\n Lowest 5 - ")
        top5 = data_pivot_total.sort_values(by = category , ascending = True)[:5]

        i = 1
        for c, v in zip(top5.index, top5[category].values):
            test = ''
            test = str(round(v,3))
            if normalization != "none":
                test = str(round(v*100.0,3))+'%'
            print(str(i) + ". " + str(c) + " - " + test)
            i += 1
    
    plt.savefig("./default/"+category+'_'+str(year)+'.png')

def line_graph_vietnam(data):
    """
    Analyzes the economy of vietnam over the existence of modern vietnam
    """
    country = "Vietnam"
    category = "net(exports-imports)"

    plt.figure()
    plt.plot(data[data["country"] == country]["year"],data[data["country"] == country][category]/(data[data["country"] == country]["imports"]+data[data["country"] == country]["exports"]))
    plt.title(country + ' ' + category +" vs. Time")
    plt.xlabel("Year")
    plt.hlines(0,1960,2014)
    plt.vlines(1964,-1,-0.9,colors='r')
    plt.vlines(1973,-1,-0.9,colors='r')
    plt.ylim(-1,0.25)
    plt.ylabel(category + " Normalized")
    plt.annotate("1964 Gulf of Tonkin\n Resolution",xy=(1964,-.95),
                xytext=(1965,-0.45),
                arrowprops = dict(facecolor='black',shrink=0.1))
    plt.annotate("1973 Paris Peace Accords",xy=(1973,-.98),
                xytext=(1980,-0.97),
                arrowprops = dict(facecolor='black',shrink=0.1))

def line_graph_ukraine(data):
    """
    Analyzes the economy of Ukraine from the fall of the USSR to 2014
    """
    country = "Ukraine"
    category = "net(exports-imports)"
    plt.figure()
    plt.plot(data[data["country"] == country]["year"],data[data["country"] == country][category]/(data[data["country"] == country]["imports"]+data[data["country"] == country]["exports"]))
    plt.title(country + ' ' + category +" vs. Time")
    plt.xlabel("Year")
    plt.hlines(0,1992,2014)
    plt.ylabel(category + " Normalized")

def top_cinc_score_countries(data):
    data_CHN=data[data['ccode']==710]
    data_USA=data[data['ccode']==2]
    data_IND=data[data['ccode']==750]
    data_RUS=data[data['ccode']==365]
    data_JPN=data[data['ccode']==740]
    data_ROK=data[data['ccode']==732]
    # Plot annually CINC score for the 6 coutries
    plt.plot(data_CHN['year'],data_CHN['cinc'],color='red')
    plt.plot(data_USA['year'],data_USA['cinc'],color='blue')
    plt.plot(data_IND['year'],data_IND['cinc'],color='orange')
    plt.plot(data_RUS['year'],data_RUS['cinc'],color='green')
    plt.plot(data_JPN['year'],data_JPN['cinc'],color='purple')
    plt.plot(data_ROK['year'],data_ROK['cinc'],color='black')
    plt.title("Top Six countries' annually CINC score")
    plt.xlabel('Year\n(1816 - 2016)')
    plt.ylabel('CINC score')
    plt.legend(['China', 'USA', 'India', 'Russia', 'Japan', 'S. Korea'], bbox_to_anchor=(1, 1), ncol=1);

def chinese_urban_population(data):
    """
    Visual of the chinese urban population over a 100 year timespan
    """
    data_CHN=data[data['ccode']==710]
    plt.plot(data_CHN['year'],data_CHN['upop'],color='red')
    plt.title("Chinese Urban Population")
    plt.xlabel('Year\n(1816 - 2016)')
    plt.ylabel('Urban Population\n(thousands)')
    plt.legend(['China']);

def current_cinc(values):
    cinc=values['cinc']
    s=f'CINC : {cinc}'
    return {'x': .85, 'y': .2, 's': s, 'ha': 'right', 'size': 11}

def show(data,country):
  plt.style.use('ggplot')
  cdata=data[data['stateabb']==country]
  cdata['year']=pd.to_datetime(cdata['year'],format="%Y")
  cdata.set_index('year', inplace=True)
  cdata=cdata.drop('ccode', axis=1)
  cdata=cdata.drop('version',axis=1)
  cdata=cdata.drop('stateabb',axis=1)
  cdata['milex']=cdata['milex']/100
  cdata.fillna(0).tail(n=10).plot_animated(
    'country.gif',  
    period_fmt="%Y",  
    title=f'Six indicators of {country}',  
    #perpendicular_bar_func='mean',  #mean
    period_summary_func=current_cinc,  #Annual CINC
    cmap='Set1',  
    n_visible=6,  
    orientation='h',
  )
