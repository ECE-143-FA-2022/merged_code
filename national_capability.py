import matplotlib.pyplot as plt
import pandas as pd
import pandas_alive


def top_cinc_score_countries(data):
    data_CHN = data[data["ccode"] == 710]
    data_USA = data[data["ccode"] == 2]
    data_IND = data[data["ccode"] == 750]
    data_RUS = data[data["ccode"] == 365]
    data_JPN = data[data["ccode"] == 740]
    data_ROK = data[data["ccode"] == 732]
    # Plot annually CINC score for the 6 coutries
    plt.plot(data_CHN["year"], data_CHN["cinc"], color="red")
    plt.plot(data_USA["year"], data_USA["cinc"], color="blue")
    plt.plot(data_IND["year"], data_IND["cinc"], color="orange")
    plt.plot(data_RUS["year"], data_RUS["cinc"], color="green")
    plt.plot(data_JPN["year"], data_JPN["cinc"], color="purple")
    plt.plot(data_ROK["year"], data_ROK["cinc"], color="black")
    plt.title("Top Six countries' annually CINC score")
    plt.xlabel("Year\n(1816 - 2016)")
    plt.ylabel("CINC score")
    plt.legend(
        ["China", "USA", "India", "Russia", "Japan", "S. Korea"],
        bbox_to_anchor=(1, 1),
        ncol=1,
    )


def chinese_urban_population(data):
    """
    Visual of the chinese urban population over a 100 year timespan
    """
    data_CHN = data[data["ccode"] == 710]
    plt.plot(data_CHN["year"], data_CHN["upop"], color="red")
    plt.title("Chinese Urban Population")
    plt.xlabel("Year\n(1816 - 2016)")
    plt.ylabel("Urban Population\n(thousands)")
    plt.legend(["China"])


def current_cinc(values):
    cinc = values["cinc"]
    s = f"CINC : {cinc}"
    return {"x": 0.85, "y": 0.2, "s": s, "ha": "right", "size": 11}


def show(data, country):
    plt.style.use("ggplot")
    cdata = data[data["stateabb"] == country]
    cdata["year"] = pd.to_datetime(cdata["year"], format="%Y")
    cdata.set_index("year", inplace=True)
    cdata = cdata.drop("ccode", axis=1)
    cdata = cdata.drop("version", axis=1)
    cdata = cdata.drop("stateabb", axis=1)
    cdata["milex"] = cdata["milex"] / 100
    cdata.fillna(0).tail(n=10).plot_animated(
        "country.gif",
        period_fmt="%Y",
        title=f"Six indicators of {country}",
        # perpendicular_bar_func='mean',  #mean
        period_summary_func=current_cinc,  # Annual CINC
        cmap="Set1",
        n_visible=6,
        orientation="h",
    )
