# A-Neural-Network-Approach-to-the-Environmental-Kuznets-Curve
Python code and data used to reproduce the empirical results of the paper *A neural network approach to the environmental Kuznets curve* by Mikkel Bennedsen, Eric Hillebrand, and Sebastian Jensen, 2023. [doi.org/10.1016/j.eneco.2023.106985](https://doi.org/10.1016/j.eneco.2023.106985).

The code is built with python 3.x (tested with version 3.6.10) and TensorFlow 2.x (tested with version 2.2.0).


## References
* Bennedsen, M., Hillebrand, E., & Jensen, S. (2023). A neural network approach to the environmental Kuznets curve. Energy Economics, 126, Article 106985. [https://doi.org/10.1016/j.eneco.2023.106985](https://doi.org/10.1016/j.eneco.2023.106985).


## Project status
Work in progress.


## Installation
Not yet available through the Python Package Index (PyPi).


## Usage
Save this repository locally (e.g. using Github Desktop), and make sure to have installed all required dependencies. 

The ***Papers*** folder contains the data and source code needed to reproduce all results of the two reference papers.

The ***Functions*** folder contains custom functions used for data preparation and implementation of the neural network-based panel data methodology: 

* The source code for the static neural network model is available in _Static_NN_model.py_.

* The source code for the dynamic neural network model is available in _Dyanmic_NN_model.py_.

 * The source code for the path-dependent neural network model is available in _Path_dependent_NN_model.py_.


### Example

*x_train* and *y_train* must be specified as dictionaries where the key-value pairs are region names and pandas DataFrames of observations, respectively.

Missing values are accounted for, but *x_train* and *y_train* must be aligned to have the same missing values. 


```python
from Dynamic_NN_model import dynamic_model as Model

print('hello_world')
```


<html>
<head>
<title>test.ipynb</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<style type="text/css">
.s0 { color: #808080;}
.s1 { color: #cc7832;}
.s2 { color: #a9b7c6;}
.s3 { color: #6a8759;}
.s4 { color: #6897bb;}
.ls0 { height: 1px; border-width: 0; color: #4d4d4d; background-color:#4d4d4d}
</style>
</head>
<body bgcolor="#2b2b2b">
<table CELLSPACING=0 CELLPADDING=5 COLS=1 WIDTH="100%" BGCOLOR="#606060" >
<tr><td><center>
<font face="Arial, Helvetica" color="#000000">
test.ipynb</font>
</center></td></tr></table>
<pre><span class="s0">#%% Importing libraries 
</span><span class="s1">import </span><span class="s2">os; os.environ[</span><span class="s3">'PYTHONHASHSEED'</span><span class="s2">] = str(</span><span class="s4">0</span><span class="s2">)</span>
<span class="s1">import </span><span class="s2">sys; sys.path.append(</span><span class="s3">'Functions'</span><span class="s2">)</span>

<span class="s1">import </span><span class="s2">numpy </span><span class="s1">as </span><span class="s2">np</span>
<span class="s1">import </span><span class="s2">tensorflow </span><span class="s1">as </span><span class="s2">tf</span>
<span class="s1">import </span><span class="s2">pandas </span><span class="s1">as </span><span class="s2">pd</span>

<span class="s1">from </span><span class="s2">fPrepare </span><span class="s1">import </span><span class="s2">fPrepare</span>
<hr class="ls0"><span class="s0">#%% Setting choice parameters 
</span><span class="s2">specification = </span><span class="s3">'dynamic'  </span><span class="s0"># Must be 'static' or 'dynamic'</span>
<span class="s2">formulation = </span><span class="s3">'regional'   </span><span class="s0"># Must be 'global', 'regional', or 'national'</span>
<span class="s2">nodes = (</span><span class="s4">8</span><span class="s1">, </span><span class="s4">8</span><span class="s1">, </span><span class="s4">8</span><span class="s2">)          </span><span class="s0"># Must be (x,), (x,y), or (x,y,z)</span>
<hr class="ls0"><span class="s0">#%% Loading model class 
</span><span class="s1">if </span><span class="s2">specification == </span><span class="s3">'static'</span><span class="s2">:</span>
    <span class="s1">from </span><span class="s2">Static_NN_model </span><span class="s1">import </span><span class="s2">static_model </span><span class="s1">as </span><span class="s2">Model</span>

<span class="s1">elif </span><span class="s2">specification == </span><span class="s3">'dynamic'</span><span class="s2">:</span>
    <span class="s1">from </span><span class="s2">Dynamic_NN_model </span><span class="s1">import </span><span class="s2">dynamic_model </span><span class="s1">as </span><span class="s2">Model</span>
<hr class="ls0"><span class="s0">#%% Loading data 
</span><span class="s2">GDP = pd.read_excel(</span><span class="s3">'Data/GDP.xlsx'</span><span class="s1">, </span><span class="s2">sheet_name=</span><span class="s3">'Python'</span><span class="s1">, </span><span class="s2">index_col=</span><span class="s4">0</span><span class="s2">)</span>
<span class="s2">GDP.sort_index(axis=</span><span class="s4">1</span><span class="s1">, </span><span class="s2">inplace=</span><span class="s1">True</span><span class="s2">)</span>

<span class="s2">POP = pd.read_excel(</span><span class="s3">'Data/Population.xlsx'</span><span class="s1">, </span><span class="s2">sheet_name=</span><span class="s3">'Python'</span><span class="s1">, </span><span class="s2">index_col=</span><span class="s4">0</span><span class="s2">) / </span><span class="s4">1e6</span>
<span class="s2">POP.sort_index(axis=</span><span class="s4">1</span><span class="s1">, </span><span class="s2">inplace=</span><span class="s1">True</span><span class="s2">)</span>

<span class="s2">DEF = pd.read_excel(</span><span class="s3">'Data/Deflator.xlsx'</span><span class="s1">, </span><span class="s2">sheet_name=</span><span class="s3">'Python'</span><span class="s1">, </span><span class="s2">index_col=</span><span class="s4">0</span><span class="s2">)</span>
<span class="s2">DEF.sort_index(axis=</span><span class="s4">1</span><span class="s1">, </span><span class="s2">inplace=</span><span class="s1">True</span><span class="s2">)</span>

<span class="s2">PPP = pd.read_excel(</span><span class="s3">'Data/PPP.xlsx'</span><span class="s1">, </span><span class="s2">sheet_name=</span><span class="s3">'Python'</span><span class="s1">, </span><span class="s2">index_col=</span><span class="s4">0</span><span class="s2">)</span>
<span class="s2">PPP.sort_index(axis=</span><span class="s4">1</span><span class="s1">, </span><span class="s2">inplace=</span><span class="s1">True</span><span class="s2">)</span>

<span class="s2">GHG = pd.read_excel(</span><span class="s3">'Data/CO2_GCP.xlsx'</span><span class="s1">, </span><span class="s2">sheet_name=</span><span class="s3">'Python'</span><span class="s1">, </span><span class="s2">index_col=</span><span class="s4">0</span><span class="s2">) * </span><span class="s4">3.664</span>
<span class="s2">GHG.sort_index(axis=</span><span class="s4">1</span><span class="s1">, </span><span class="s2">inplace=</span><span class="s1">True</span><span class="s2">)</span>
<hr class="ls0"><span class="s0">#%% Preparing data 
</span><span class="s2">gdp_est</span><span class="s1">, </span><span class="s2">ghg_est</span><span class="s1">, </span><span class="s2">pop_est = fPrepare(GDP</span><span class="s1">, </span><span class="s2">POP</span><span class="s1">, </span><span class="s2">DEF</span><span class="s1">, </span><span class="s2">PPP</span><span class="s1">, </span><span class="s2">GHG)</span>
<hr class="ls0"><span class="s0">#%% 
</span><span class="s2">model = Model(nodes=nodes</span><span class="s1">, </span><span class="s2">x_train=gdp_est</span><span class="s1">, </span><span class="s2">y_train=ghg_est</span><span class="s1">, </span><span class="s2">pop_train=pop_est</span><span class="s1">, </span><span class="s2">formulation=formulation)</span>

<span class="s2">model.fit(lr=</span><span class="s4">0.001</span><span class="s1">, </span><span class="s2">min_delta=</span><span class="s4">1e-6</span><span class="s1">, </span><span class="s2">patience=</span><span class="s4">100</span><span class="s1">, </span><span class="s2">verbose=</span><span class="s1">False</span><span class="s2">)</span>

<span class="s2">model.in_sample_predictions()</span>
<span class="s2">BIC = model.BIC</span>
<span class="s2">print(BIC)</span>
</pre>
</body>
</html>


## License
Not yet licensed.


## Dependencies
numpy (tested with version 1.19.4)

scipy (tested with version 1.4.1)

pandas (tested with version 1.1.1)

tensorflow (tested with version 2.2.0)

keras (tested with version 2.4.3)

matplotlib (tested with version 3.1.1)

seaborn (tested with version 0.9.0)


## Contact
Sebastian Jensen

Mail: smjensen@econ.au.dk

Website: https://sites.google.com/view/sebastianjensen

GitHub profile: https://github.com/Sebastian-Jensen

Project link: https://github.com/Sebastian-Jensen/A-Neural-Network-Approach-to-the-Environmental-Kuznets-Curve


## Co-authors
[Eric Hillebrand](https://sites.google.com/site/erichillebrand)

[Mikkel Bennedsen](https://sites.google.com/site/mbennedsen/home)
