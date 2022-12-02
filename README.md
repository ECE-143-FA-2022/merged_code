# Analyzing World-Wide War Trends with Python
### File structure
* Datasets used:
  ```
  version4.1_csv, COW-country-codes.csv, Diplomatic_Exchange_2006v1.csv, MIDLOCA_2.1.csv, NMC-60-abridged.csv, National_Import_Export.csv, dyadic_mid_4.02.csv
  ```
* requirement.txt<br>
Includes all the third-party packages required.

* Final_Presentation_Slides.pdf<br>
The pdf file of our final slides.

* functions.py
Code for presentations, imported into war_trends.ipynb

### Environment setup
The Python version we use is `3.8.5`.

Install Jupyter notebook:
```
pip install jupyterlab==3.3.2
pip install notebook==6.4.10
```

Install required packages:<br>
&emsp;&emsp;matplotlib==3.6.2<br>
&emsp;&emsp;pandas==1.4.4<br>
&emsp;&emsp;numpy==1.23.1<br>
&emsp;&emsp;thefuzz==0.19.0<br>
&emsp;&emsp;ipywidgets==8.0.2<br>
&emsp;&emsp;plotly==5.11.0<br>
&emsp;&emsp;kaleido==0.2.1<br>
&emsp;&emsp;seaborn==0.12.1<br>
&emsp;&emsp;geopandas==0.12.1<br>
&emsp;&emsp;mpl-chord-diagram==0.4.0<br>
&emsp;&emsp;pandas_alive==0.2.3<br>
```
pip install -r requirements.txt
```

### Generating plots
Run the nodebook `war_trends.ipynb` to plot world-wide war trends. Note that to see the interactive plots in the notebook, one needs to rerun it.

### Citations
The datasets were downloaded from [Correlates of War](https://correlatesofwar.org/data-sets/).

```
Gibler, Douglas M. 2009. International military alliances, 1648-2008. CQ Press.  

Singer, J. David, and Melvin Small. 1966. “Formal Alliances, 1815-1939.” Journal of Peace Research 3:1-31.

Small, Melvin, and J. David Singer. 1969. “Formal Alliances, 1815-1965: An Extension of the Basic Data.” Journal of Peace Research 6:257-282.

Barbieri, Katherine and Omar M. G. Omar Keshk. 2016. Correlates of War Project Trade Data Set Codebook, Version 4.0. Online: https://correlatesofwar.org.

Barbieri, Katherine, Omar M. G. Keshk, and Brian Pollins. 2009. “TRADING DATA: Evaluating our Assumptions and Coding Rules.” Conflict Management and Peace Science. 26(5): 471-491.

Singer, J. David, Stuart Bremer, and John Stuckey. (1972). “Capability Distribution, Uncertainty, and Major Power War, 1820-1965.” in Bruce Russett (ed) Peace, War, and Numbers, Beverly Hills: Sage, 19-48.

Braithwaite, A. 2010. “MIDLOC: Introducing the Militarized Interstate Dispute (MID) Location Dataset.” Journal of Peace Research 47(1): 91-98.

Bezerra, P., & Braithwaite, A. 2019. Codebook for the Militarized Interstate Dispute Location (MIDLOC-A/I) Dataset, v2.1.
```
