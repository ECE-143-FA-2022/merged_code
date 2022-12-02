import numpy as np
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
import itertools
from collections import defaultdict
from mpl_chord_diagram import chord_diagram


def sankey_new(data, name_to_code, year, coun_list):
    """
    Produces graph model in the form of a sankey diagram
    nodes exist on the outer perimiter of a circle
    edges are inside of said circle
    """

    list1 = list(coun_list)
    coun_list = [name_to_code[n] for n in list1]

    countries = defaultdict(int)
    coun_name = []
    for i in range(len(coun_list)):
        for idx in range(len(data)):

            if data[idx][1] == coun_list[i]:
                coun_name.append(data[idx][0])
                break

    data1 = pd.read_csv("Diplomatic_Exchange_2006v1.csv", sep=",")
    data1 = data1.values.tolist()

    exchange_in_year = defaultdict(int)
    for idx in range(len(data1)):
        if (int(data1[idx][0]), int(data1[idx][1])) in itertools.permutations(
            coun_list, 2
        ):
            if int(data1[idx][2]) >= int(year) and int(data1[idx][2]) < int(year) + 10:
                exchange_in_year[(data1[idx][0], data1[idx][1])] += data1[idx][4]

    exchange = np.zeros((len(coun_list), len(coun_list)))
    for i in range(len(coun_list)):

        for j in range(len(coun_list)):

            exchange[i][j] = exchange_in_year[(coun_list[i], coun_list[j])]
    Num = len(coun_list)
    N1 = Num
    f = 10 - int(Num / 10)
    chord_diagram(exchange, coun_name, fontsize=f)
