import pandas as pd
import seaborn as sns

sns.set_style(
    "whitegrid",
    {
        "axes.axisbelow": False,
        "grid.color": "w",
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    },
)
import matplotlib.pyplot as plt
import plotly.express as px


def plotLocations(data2):
    """
    Plots the locations of military interstate disputes
    :param data2: COW interstate disputes dataset
    :type data2: Pandas Dataframe
    """

    fig = px.scatter_geo(
        data2,
        lon="longitude",
        lat="latitude",
        color="precision",
        hover_name="countries",
        animation_frame="year",
        color_continuous_scale=px.colors.sequential.Bluered,
        basemap_visible=True,
        title="Total number of disputes started in each year",
    )

    fig.update_geos(
        projection_type="mercator",
        showcountries=True,
        countrycolor="Black",
        lataxis_range=[-30, 86],
    )
    fig.update_layout(
        width=900,
        height=700,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
    )
    return fig


def barTotalDisputesUpto(data, year):

    # plotting countries with most involvements in disputes
    data_for_years = data[data["year"] <= year]
    disputes_namea = pd.DataFrame(data_for_years["namea"].value_counts())
    top_10_countries = disputes_namea.head(10)
    xaxis = list(top_10_countries.index)
    yaxis = top_10_countries["namea"]
    fig = sns.barplot(x=xaxis, y=yaxis, palette="Spectral")
    plt.title("Countries involved in most disputes from 1816 to " + str(year))
    plt.xlabel("Countries")
    plt.ylabel("Number of Disputes")
    plt.show(fig)
