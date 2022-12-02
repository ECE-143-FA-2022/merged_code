import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from thefuzz import fuzz


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
    m = re.search("[a-zA-Z]", x)
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
        self.countries_data = pd.read_csv(
            "./version4.1_csv/cow.txt", sep=";", skiprows=28
        )
        self.countries_data.BGN_proper = self.countries_data.BGN_proper.apply(
            lambda x: x.strip(" ")
        )
        self.countries_data.BGN_name = self.countries_data.BGN_name.apply(
            lambda x: x.strip(" ")
        )
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
        country_label_obj = self.scatter_points(
            self.countries_data, "longitude", "latitude", "BGN_proper", "country/region"
        )
        self.fig.add_trace(country_label_obj)

    @staticmethod
    def scatter_points(df, lon_key, lat_key, text_key, name=None):
        graph_object = go.Scattergeo(
            lon=df[lon_key],
            lat=df[lat_key],
            hoverinfo="text",
            text=df[text_key],
            mode="markers",
            marker=dict(
                size=2,
                color="rgb(255, 0, 0)",
                line=dict(width=3, color="rgba(68, 68, 68, 0)"),
            ),
            name=name,
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
        df["start_lon"] = df["state_name1"].apply(self.get_longitude)
        df["end_lon"] = df["state_name2"].apply(self.get_longitude)
        df["start_lat"] = df["state_name1"].apply(self.get_latitude)
        df["end_lat"] = df["state_name2"].apply(self.get_latitude)
        df = df.loc[df["start_lon"].notnull() & df["end_lon"].notnull()]
        self._connect(df.loc[df["defense"] == 1], "rgb(255,0,0)", name="defense")
        self._connect(
            df.loc[(df["defense"] == 0) & (df["neutrality"] == 1)],
            "rgb(0, 255, 0)",
            name="neutrality",
        )
        self._connect(
            df.loc[
                (df["defense"] == 0)
                & (df["neutrality"] == 0)
                & (df["nonaggression"] == 1)
            ],
            "rgb(0, 0, 255)",
            name="nonaggression",
        )
        self._connect(
            df.loc[
                (df["defense"] == 0)
                & (df["neutrality"] == 0)
                & (df["nonaggression"] == 0)
                & df["entente"]
                == 1
            ],
            "rgb(255,165,0)",
            name="entente",
        )
        return len(df)

    def _connect(self, df, color="#0099C6", name=None):
        lons = np.empty(3 * len(df))
        lons[::3] = df["start_lon"]
        lons[1::3] = df["end_lon"]
        lons[2::3] = None
        lats = np.empty(3 * len(df))
        lats[::3] = df["start_lat"]
        lats[1::3] = df["end_lat"]
        lats[2::3] = None
        if len(df) == 0:
            lons = np.array([None])
            lats = np.array([None])

        self.fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                hoverinfo="none",
                line=dict(width=1.2, color=color),
                opacity=0.9,
                name=name,
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
        lataxis_range=[-30, 86],
    )
    alliance_map.fig.update_layout(
        title=str(year),
        width=900,
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        #         showlegend=False,
    )
    alliance_map.fig.write_image(f"./plotly2/{year}.png", scale=1.5)
