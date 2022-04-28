# Website
[Click Here](https://ramayyala.github.io/TRGN_515_Final_Project/)
# Libraries


```python
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import GenericUnivariateSelect
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
```


```python
column_names=[
    "ID", # Id of patients
    "Age", # 18-24, 25-34, 35-44, 45-54, 55-64, 65+
    "Gender", # Male,Female
    "Educ", # Left before age 16, left @ 16, @ 17, @ 18, some college, prof cert, univ degree, masters, doctorate
    "Cntry", # Country: AUS, CAN, NZ, Other, IRE, UK, USA
    "Ethn", # Ethnicity: Asian, Black, Mixed Bla/As, Mixed Whi/As, Mixed Whi/Bla, Other
    "NS", #Neuroticism Score
    "ES",  # Extroversion Score
    "OS", #Openness to experience Score
    "AS",  # Agreeableness Score
    "CS",  # Conscientiousness Score
    "Imp", # Impulsivity, Lickert scale with -3 = least impulsive, +3 = most impulsive
    "SS",  # Sensation seeking, part of the Impulsiveness assessment, -3 < score > +3
    "Alcohol", # Class of alcohol consumption
    "Amphet", # Class of amphetamine consumption
    "Amyl", # Class of amyl nitrite consumption
    "Benzos", # Class of benzodiazepine consumption
    "Caffeine", # Class of caffeine consumption
    "Cannabis", # Class of cannabis consumption
    "Choco", # Class of chocolate consumption
    "Coke", # Class of cocaine consumption
    "Crack", # Class of crack cocaine consumption
    "Ecstasy", # Class of ecstasy consumption
    "Heroin", # Class of heroin consumption
    "Ketamine", # Class of ketamine consumption
    "LegalH", # Class of legal highs consumption
    "LSD", # Class of LSD consumption
    "Meth", # Class of methamphetamine consumption
    "Shrooms", # Class of mushrooms consumption
    "Nicotine", # Class of nicotine consumption
    "Semer",# Class of fictitious drug Semeron consumption
    "VSA" # Class of volatile substance abuse consumption
]
df=pd.read_csv("drug_consumption.data",sep=",", header=None, names=column_names)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Educ</th>
      <th>Cntry</th>
      <th>Ethn</th>
      <th>NS</th>
      <th>ES</th>
      <th>OS</th>
      <th>AS</th>
      <th>...</th>
      <th>Ecstasy</th>
      <th>Heroin</th>
      <th>Ketamine</th>
      <th>LegalH</th>
      <th>LSD</th>
      <th>Meth</th>
      <th>Shrooms</th>
      <th>Nicotine</th>
      <th>Semer</th>
      <th>VSA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.49788</td>
      <td>0.48246</td>
      <td>-0.05921</td>
      <td>0.96082</td>
      <td>0.12600</td>
      <td>0.31287</td>
      <td>-0.57545</td>
      <td>-0.58331</td>
      <td>-0.91699</td>
      <td>...</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL0</td>
      <td>CL0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.07854</td>
      <td>-0.48246</td>
      <td>1.98437</td>
      <td>0.96082</td>
      <td>-0.31685</td>
      <td>-0.67825</td>
      <td>1.93886</td>
      <td>1.43533</td>
      <td>0.76096</td>
      <td>...</td>
      <td>CL4</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL3</td>
      <td>CL0</td>
      <td>CL4</td>
      <td>CL0</td>
      <td>CL0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.49788</td>
      <td>-0.48246</td>
      <td>-0.05921</td>
      <td>0.96082</td>
      <td>-0.31685</td>
      <td>-0.46725</td>
      <td>0.80523</td>
      <td>-0.84732</td>
      <td>-1.62090</td>
      <td>...</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL1</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-0.95197</td>
      <td>0.48246</td>
      <td>1.16365</td>
      <td>0.96082</td>
      <td>-0.31685</td>
      <td>-0.14882</td>
      <td>-0.80615</td>
      <td>-0.01928</td>
      <td>0.59042</td>
      <td>...</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL0</td>
      <td>CL0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.49788</td>
      <td>0.48246</td>
      <td>1.98437</td>
      <td>0.96082</td>
      <td>-0.31685</td>
      <td>0.73545</td>
      <td>-1.63340</td>
      <td>-0.45174</td>
      <td>-0.30172</td>
      <td>...</td>
      <td>CL1</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL1</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL2</td>
      <td>CL0</td>
      <td>CL0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>1884</td>
      <td>-0.95197</td>
      <td>0.48246</td>
      <td>-0.61113</td>
      <td>-0.57009</td>
      <td>-0.31685</td>
      <td>-1.19430</td>
      <td>1.74091</td>
      <td>1.88511</td>
      <td>0.76096</td>
      <td>...</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL3</td>
      <td>CL3</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL5</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>1885</td>
      <td>-0.95197</td>
      <td>-0.48246</td>
      <td>-0.61113</td>
      <td>-0.57009</td>
      <td>-0.31685</td>
      <td>-0.24649</td>
      <td>1.74091</td>
      <td>0.58331</td>
      <td>0.76096</td>
      <td>...</td>
      <td>CL2</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL3</td>
      <td>CL5</td>
      <td>CL4</td>
      <td>CL4</td>
      <td>CL5</td>
      <td>CL0</td>
      <td>CL0</td>
    </tr>
    <tr>
      <th>1882</th>
      <td>1886</td>
      <td>-0.07854</td>
      <td>0.48246</td>
      <td>0.45468</td>
      <td>-0.57009</td>
      <td>-0.31685</td>
      <td>1.13281</td>
      <td>-1.37639</td>
      <td>-1.27553</td>
      <td>-1.77200</td>
      <td>...</td>
      <td>CL4</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL0</td>
      <td>CL2</td>
      <td>CL6</td>
      <td>CL0</td>
      <td>CL0</td>
    </tr>
    <tr>
      <th>1883</th>
      <td>1887</td>
      <td>-0.95197</td>
      <td>0.48246</td>
      <td>-0.61113</td>
      <td>-0.57009</td>
      <td>-0.31685</td>
      <td>0.91093</td>
      <td>-1.92173</td>
      <td>0.29338</td>
      <td>-1.62090</td>
      <td>...</td>
      <td>CL3</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL3</td>
      <td>CL3</td>
      <td>CL0</td>
      <td>CL3</td>
      <td>CL4</td>
      <td>CL0</td>
      <td>CL0</td>
    </tr>
    <tr>
      <th>1884</th>
      <td>1888</td>
      <td>-0.95197</td>
      <td>-0.48246</td>
      <td>-0.61113</td>
      <td>0.21128</td>
      <td>-0.31685</td>
      <td>-0.46725</td>
      <td>2.12700</td>
      <td>1.65653</td>
      <td>1.11406</td>
      <td>...</td>
      <td>CL3</td>
      <td>CL0</td>
      <td>CL0</td>
      <td>CL3</td>
      <td>CL3</td>
      <td>CL0</td>
      <td>CL3</td>
      <td>CL6</td>
      <td>CL0</td>
      <td>CL2</td>
    </tr>
  </tbody>
</table>
<p>1885 rows × 32 columns</p>
</div>



# EDA


```python
df.shape
```




    (1885, 32)




```python
df.columns.values
```




    array(['ID', 'Age', 'Gender', 'Educ', 'Cntry', 'Ethn', 'NS', 'ES', 'OS',
           'AS', 'CS', 'Imp', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos',
           'Caffeine', 'Cannabis', 'Choco', 'Coke', 'Crack', 'Ecstasy',
           'Heroin', 'Ketamine', 'LegalH', 'LSD', 'Meth', 'Shrooms',
           'Nicotine', 'Semer', 'VSA'], dtype=object)




```python
df.isnull().sum()
```




    ID          0
    Age         0
    Gender      0
    Educ        0
    Cntry       0
    Ethn        0
    NS          0
    ES          0
    OS          0
    AS          0
    CS          0
    Imp         0
    SS          0
    Alcohol     0
    Amphet      0
    Amyl        0
    Benzos      0
    Caffeine    0
    Cannabis    0
    Choco       0
    Coke        0
    Crack       0
    Ecstasy     0
    Heroin      0
    Ketamine    0
    LegalH      0
    LSD         0
    Meth        0
    Shrooms     0
    Nicotine    0
    Semer       0
    VSA         0
    dtype: int64




```python
df.describe()
# In this case, no drug use classes are shown as they are still encoded as text, not numbers which we will change later 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Educ</th>
      <th>Cntry</th>
      <th>Ethn</th>
      <th>NS</th>
      <th>ES</th>
      <th>OS</th>
      <th>AS</th>
      <th>CS</th>
      <th>Imp</th>
      <th>SS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1885.000000</td>
      <td>1885.00000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
      <td>1885.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>945.294960</td>
      <td>0.03461</td>
      <td>-0.000256</td>
      <td>-0.003806</td>
      <td>0.355542</td>
      <td>-0.309577</td>
      <td>0.000047</td>
      <td>-0.000163</td>
      <td>-0.000534</td>
      <td>-0.000245</td>
      <td>-0.000386</td>
      <td>0.007216</td>
      <td>-0.003292</td>
    </tr>
    <tr>
      <th>std</th>
      <td>545.167641</td>
      <td>0.87836</td>
      <td>0.482588</td>
      <td>0.950078</td>
      <td>0.700335</td>
      <td>0.166226</td>
      <td>0.998106</td>
      <td>0.997448</td>
      <td>0.996229</td>
      <td>0.997440</td>
      <td>0.997523</td>
      <td>0.954435</td>
      <td>0.963701</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-0.95197</td>
      <td>-0.482460</td>
      <td>-2.435910</td>
      <td>-0.570090</td>
      <td>-1.107020</td>
      <td>-3.464360</td>
      <td>-3.273930</td>
      <td>-3.273930</td>
      <td>-3.464360</td>
      <td>-3.464360</td>
      <td>-2.555240</td>
      <td>-2.078480</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>474.000000</td>
      <td>-0.95197</td>
      <td>-0.482460</td>
      <td>-0.611130</td>
      <td>-0.570090</td>
      <td>-0.316850</td>
      <td>-0.678250</td>
      <td>-0.695090</td>
      <td>-0.717270</td>
      <td>-0.606330</td>
      <td>-0.652530</td>
      <td>-0.711260</td>
      <td>-0.525930</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>946.000000</td>
      <td>-0.07854</td>
      <td>-0.482460</td>
      <td>-0.059210</td>
      <td>0.960820</td>
      <td>-0.316850</td>
      <td>0.042570</td>
      <td>0.003320</td>
      <td>-0.019280</td>
      <td>-0.017290</td>
      <td>-0.006650</td>
      <td>-0.217120</td>
      <td>0.079870</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1417.000000</td>
      <td>0.49788</td>
      <td>0.482460</td>
      <td>0.454680</td>
      <td>0.960820</td>
      <td>-0.316850</td>
      <td>0.629670</td>
      <td>0.637790</td>
      <td>0.723300</td>
      <td>0.760960</td>
      <td>0.584890</td>
      <td>0.529750</td>
      <td>0.765400</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1888.000000</td>
      <td>2.59171</td>
      <td>0.482460</td>
      <td>1.984370</td>
      <td>0.960820</td>
      <td>1.907250</td>
      <td>3.273930</td>
      <td>3.273930</td>
      <td>2.901610</td>
      <td>3.464360</td>
      <td>3.464360</td>
      <td>2.901610</td>
      <td>1.921730</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    ID            int64
    Age         float64
    Gender      float64
    Educ        float64
    Cntry       float64
    Ethn        float64
    NS          float64
    ES          float64
    OS          float64
    AS          float64
    CS          float64
    Imp         float64
    SS          float64
    Alcohol      object
    Amphet       object
    Amyl         object
    Benzos       object
    Caffeine     object
    Cannabis     object
    Choco        object
    Coke         object
    Crack        object
    Ecstasy      object
    Heroin       object
    Ketamine     object
    LegalH       object
    LSD          object
    Meth         object
    Shrooms      object
    Nicotine     object
    Semer        object
    VSA          object
    dtype: object




```python
df.hist(figsize = (15,15))
```




    array([[<AxesSubplot:title={'center':'ID'}>,
            <AxesSubplot:title={'center':'Age'}>,
            <AxesSubplot:title={'center':'Gender'}>,
            <AxesSubplot:title={'center':'Educ'}>],
           [<AxesSubplot:title={'center':'Cntry'}>,
            <AxesSubplot:title={'center':'Ethn'}>,
            <AxesSubplot:title={'center':'NS'}>,
            <AxesSubplot:title={'center':'ES'}>],
           [<AxesSubplot:title={'center':'OS'}>,
            <AxesSubplot:title={'center':'AS'}>,
            <AxesSubplot:title={'center':'CS'}>,
            <AxesSubplot:title={'center':'Imp'}>],
           [<AxesSubplot:title={'center':'SS'}>, <AxesSubplot:>,
            <AxesSubplot:>, <AxesSubplot:>]], dtype=object)




    
![png](Drug_Consumption_files/Drug_Consumption_9_1.png)
    


From the plot above, we see that no drug use classes are shown as they are still encoded as text, not numbers, so we need to convert those values to numerical. In addition, it is evident that we do not have a target column for our model to use. 

## Data Fixing

### Convert Drug Classes to Numerical 

**CL0**: Never used this substance<br>
**CL1**: Used over a decade ago<br>
**CL2**: Used in last decade<br>
**CL3**: Used in last year<br>
**CL4**: Used in last month<br>
**CL5**: Used in last week<br>
**CL6**: Used in last day<br>


```python

#df.replace(["CL0","CL1","CL2","CL3","CL4","CL5","CL6"],[-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0],inplace=True)
df.replace(["CL0","CL1","CL2","CL3","CL4","CL5","CL6"],[0.0,1.0,2.0,3.0,4.0,5.0,6.0],inplace=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Educ</th>
      <th>Cntry</th>
      <th>Ethn</th>
      <th>NS</th>
      <th>ES</th>
      <th>OS</th>
      <th>AS</th>
      <th>...</th>
      <th>Ecstasy</th>
      <th>Heroin</th>
      <th>Ketamine</th>
      <th>LegalH</th>
      <th>LSD</th>
      <th>Meth</th>
      <th>Shrooms</th>
      <th>Nicotine</th>
      <th>Semer</th>
      <th>VSA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.49788</td>
      <td>0.48246</td>
      <td>-0.05921</td>
      <td>0.96082</td>
      <td>0.12600</td>
      <td>0.31287</td>
      <td>-0.57545</td>
      <td>-0.58331</td>
      <td>-0.91699</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.07854</td>
      <td>-0.48246</td>
      <td>1.98437</td>
      <td>0.96082</td>
      <td>-0.31685</td>
      <td>-0.67825</td>
      <td>1.93886</td>
      <td>1.43533</td>
      <td>0.76096</td>
      <td>...</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.49788</td>
      <td>-0.48246</td>
      <td>-0.05921</td>
      <td>0.96082</td>
      <td>-0.31685</td>
      <td>-0.46725</td>
      <td>0.80523</td>
      <td>-0.84732</td>
      <td>-1.62090</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-0.95197</td>
      <td>0.48246</td>
      <td>1.16365</td>
      <td>0.96082</td>
      <td>-0.31685</td>
      <td>-0.14882</td>
      <td>-0.80615</td>
      <td>-0.01928</td>
      <td>0.59042</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.49788</td>
      <td>0.48246</td>
      <td>1.98437</td>
      <td>0.96082</td>
      <td>-0.31685</td>
      <td>0.73545</td>
      <td>-1.63340</td>
      <td>-0.45174</td>
      <td>-0.30172</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>1884</td>
      <td>-0.95197</td>
      <td>0.48246</td>
      <td>-0.61113</td>
      <td>-0.57009</td>
      <td>-0.31685</td>
      <td>-1.19430</td>
      <td>1.74091</td>
      <td>1.88511</td>
      <td>0.76096</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>1885</td>
      <td>-0.95197</td>
      <td>-0.48246</td>
      <td>-0.61113</td>
      <td>-0.57009</td>
      <td>-0.31685</td>
      <td>-0.24649</td>
      <td>1.74091</td>
      <td>0.58331</td>
      <td>0.76096</td>
      <td>...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1882</th>
      <td>1886</td>
      <td>-0.07854</td>
      <td>0.48246</td>
      <td>0.45468</td>
      <td>-0.57009</td>
      <td>-0.31685</td>
      <td>1.13281</td>
      <td>-1.37639</td>
      <td>-1.27553</td>
      <td>-1.77200</td>
      <td>...</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1883</th>
      <td>1887</td>
      <td>-0.95197</td>
      <td>0.48246</td>
      <td>-0.61113</td>
      <td>-0.57009</td>
      <td>-0.31685</td>
      <td>0.91093</td>
      <td>-1.92173</td>
      <td>0.29338</td>
      <td>-1.62090</td>
      <td>...</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1884</th>
      <td>1888</td>
      <td>-0.95197</td>
      <td>-0.48246</td>
      <td>-0.61113</td>
      <td>0.21128</td>
      <td>-0.31685</td>
      <td>-0.46725</td>
      <td>2.12700</td>
      <td>1.65653</td>
      <td>1.11406</td>
      <td>...</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>1885 rows × 32 columns</p>
</div>



### Create Target Column

The Target Column named Drug Use will be a binary column that is made up of 0s and 1s. 0 means the individual is not likely to use drugs while 1 means that the individual is likely to use. 


```python
drug_names = ["Amphet","Amyl","Benzos","Cannabis","Coke","Crack","Ecstasy","Heroin","Ketamine","LegalH","LSD","Meth","Shrooms","Semer","VSA"]
df["Drug Use"]=int(0)
for i in df.index.values:
    count = -1
    while count < 14:
        count += 1
        if df.loc[i, drug_names[count]] > 1.0:
            # set the target column to 1 
            df.loc[[i],['Drug Use']] = int(1)
            break
```


```python
df.hist(figsize = (15,15))
```




    array([[<AxesSubplot:title={'center':'ID'}>,
            <AxesSubplot:title={'center':'Age'}>,
            <AxesSubplot:title={'center':'Gender'}>,
            <AxesSubplot:title={'center':'Educ'}>,
            <AxesSubplot:title={'center':'Cntry'}>,
            <AxesSubplot:title={'center':'Ethn'}>],
           [<AxesSubplot:title={'center':'NS'}>,
            <AxesSubplot:title={'center':'ES'}>,
            <AxesSubplot:title={'center':'OS'}>,
            <AxesSubplot:title={'center':'AS'}>,
            <AxesSubplot:title={'center':'CS'}>,
            <AxesSubplot:title={'center':'Imp'}>],
           [<AxesSubplot:title={'center':'SS'}>,
            <AxesSubplot:title={'center':'Alcohol'}>,
            <AxesSubplot:title={'center':'Amphet'}>,
            <AxesSubplot:title={'center':'Amyl'}>,
            <AxesSubplot:title={'center':'Benzos'}>,
            <AxesSubplot:title={'center':'Caffeine'}>],
           [<AxesSubplot:title={'center':'Cannabis'}>,
            <AxesSubplot:title={'center':'Choco'}>,
            <AxesSubplot:title={'center':'Coke'}>,
            <AxesSubplot:title={'center':'Crack'}>,
            <AxesSubplot:title={'center':'Ecstasy'}>,
            <AxesSubplot:title={'center':'Heroin'}>],
           [<AxesSubplot:title={'center':'Ketamine'}>,
            <AxesSubplot:title={'center':'LegalH'}>,
            <AxesSubplot:title={'center':'LSD'}>,
            <AxesSubplot:title={'center':'Meth'}>,
            <AxesSubplot:title={'center':'Shrooms'}>,
            <AxesSubplot:title={'center':'Nicotine'}>],
           [<AxesSubplot:title={'center':'Semer'}>,
            <AxesSubplot:title={'center':'VSA'}>,
            <AxesSubplot:title={'center':'Drug Use'}>, <AxesSubplot:>,
            <AxesSubplot:>, <AxesSubplot:>]], dtype=object)




    
![png](Drug_Consumption_files/Drug_Consumption_18_1.png)
    



```python
labels=['Not Likely to Use','Likely to Use']
t=sns.countplot(df['Drug Use'])
t.set(xticks=range(len(labels)), xticklabels=[i for i in labels])
```

    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(





    [[<matplotlib.axis.XTick at 0x1520107d070>,
      <matplotlib.axis.XTick at 0x1520107d2e0>],
     [Text(0, 0, 'Not Likely to Use'), Text(1, 0, 'Likely to Use')]]




    
![png](Drug_Consumption_files/Drug_Consumption_19_2.png)
    


From the above plot, we see that there is an imbalance in the classification of whether an individual is more likely to to use drugs than not, by a significant amount. Because of this imbalance, we must use an undersampling method when creating training and test data sets for modeling. 

## Feature Selection


```python
X = df[column_names]
drug_corr = X.corr()
drug_corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Educ</th>
      <th>Cntry</th>
      <th>Ethn</th>
      <th>NS</th>
      <th>ES</th>
      <th>OS</th>
      <th>AS</th>
      <th>...</th>
      <th>Ecstasy</th>
      <th>Heroin</th>
      <th>Ketamine</th>
      <th>LegalH</th>
      <th>LSD</th>
      <th>Meth</th>
      <th>Shrooms</th>
      <th>Nicotine</th>
      <th>Semer</th>
      <th>VSA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ID</th>
      <td>1.000000</td>
      <td>-0.271395</td>
      <td>-0.025467</td>
      <td>-0.025253</td>
      <td>-0.340751</td>
      <td>0.059309</td>
      <td>0.018639</td>
      <td>-0.046960</td>
      <td>0.173565</td>
      <td>-0.028782</td>
      <td>...</td>
      <td>0.167231</td>
      <td>0.091180</td>
      <td>0.074800</td>
      <td>0.220806</td>
      <td>0.215234</td>
      <td>0.175429</td>
      <td>0.202910</td>
      <td>0.063197</td>
      <td>0.050454</td>
      <td>0.101165</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.271395</td>
      <td>1.000000</td>
      <td>0.110286</td>
      <td>0.158811</td>
      <td>0.354241</td>
      <td>-0.069753</td>
      <td>-0.136654</td>
      <td>-0.033849</td>
      <td>-0.226778</td>
      <td>0.063504</td>
      <td>...</td>
      <td>-0.384784</td>
      <td>-0.121675</td>
      <td>-0.220689</td>
      <td>-0.419839</td>
      <td>-0.323207</td>
      <td>-0.191503</td>
      <td>-0.331456</td>
      <td>-0.248883</td>
      <td>-0.049729</td>
      <td>-0.229657</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>-0.025467</td>
      <td>0.110286</td>
      <td>1.000000</td>
      <td>0.196774</td>
      <td>0.216271</td>
      <td>-0.001213</td>
      <td>0.074646</td>
      <td>0.057864</td>
      <td>-0.131021</td>
      <td>0.219743</td>
      <td>...</td>
      <td>-0.228574</td>
      <td>-0.136728</td>
      <td>-0.189825</td>
      <td>-0.315676</td>
      <td>-0.278983</td>
      <td>-0.181489</td>
      <td>-0.272431</td>
      <td>-0.192084</td>
      <td>0.013354</td>
      <td>-0.134852</td>
    </tr>
    <tr>
      <th>Educ</th>
      <td>-0.025253</td>
      <td>0.158811</td>
      <td>0.196774</td>
      <td>1.000000</td>
      <td>0.225311</td>
      <td>-0.036099</td>
      <td>-0.100993</td>
      <td>0.115645</td>
      <td>0.057994</td>
      <td>0.091088</td>
      <td>...</td>
      <td>-0.159819</td>
      <td>-0.131051</td>
      <td>-0.076479</td>
      <td>-0.208656</td>
      <td>-0.177817</td>
      <td>-0.170103</td>
      <td>-0.169762</td>
      <td>-0.240547</td>
      <td>-0.036342</td>
      <td>-0.120540</td>
    </tr>
    <tr>
      <th>Cntry</th>
      <td>-0.340751</td>
      <td>0.354241</td>
      <td>0.216271</td>
      <td>0.225311</td>
      <td>1.000000</td>
      <td>-0.127946</td>
      <td>-0.136191</td>
      <td>0.109524</td>
      <td>-0.341969</td>
      <td>0.150921</td>
      <td>...</td>
      <td>-0.336328</td>
      <td>-0.300210</td>
      <td>-0.112577</td>
      <td>-0.426030</td>
      <td>-0.498263</td>
      <td>-0.413946</td>
      <td>-0.490052</td>
      <td>-0.277913</td>
      <td>-0.068018</td>
      <td>-0.267033</td>
    </tr>
    <tr>
      <th>Ethn</th>
      <td>0.059309</td>
      <td>-0.069753</td>
      <td>-0.001213</td>
      <td>-0.036099</td>
      <td>-0.127946</td>
      <td>1.000000</td>
      <td>0.047642</td>
      <td>0.018402</td>
      <td>0.084816</td>
      <td>-0.038726</td>
      <td>...</td>
      <td>0.071826</td>
      <td>0.042881</td>
      <td>0.031961</td>
      <td>0.077511</td>
      <td>0.129031</td>
      <td>0.063805</td>
      <td>0.115962</td>
      <td>0.077724</td>
      <td>0.022716</td>
      <td>0.087011</td>
    </tr>
    <tr>
      <th>NS</th>
      <td>0.018639</td>
      <td>-0.136654</td>
      <td>0.074646</td>
      <td>-0.100993</td>
      <td>-0.136191</td>
      <td>0.047642</td>
      <td>1.000000</td>
      <td>-0.431051</td>
      <td>0.010177</td>
      <td>-0.216964</td>
      <td>...</td>
      <td>0.069948</td>
      <td>0.172685</td>
      <td>0.062750</td>
      <td>0.113342</td>
      <td>0.037095</td>
      <td>0.184672</td>
      <td>0.042386</td>
      <td>0.128430</td>
      <td>-0.001673</td>
      <td>0.115086</td>
    </tr>
    <tr>
      <th>ES</th>
      <td>-0.046960</td>
      <td>-0.033849</td>
      <td>0.057864</td>
      <td>0.115645</td>
      <td>0.109524</td>
      <td>0.018402</td>
      <td>-0.431051</td>
      <td>1.000000</td>
      <td>0.245277</td>
      <td>0.157336</td>
      <td>...</td>
      <td>0.078822</td>
      <td>-0.079998</td>
      <td>0.018727</td>
      <td>-0.037383</td>
      <td>0.018166</td>
      <td>-0.121708</td>
      <td>0.021105</td>
      <td>-0.019196</td>
      <td>0.022909</td>
      <td>-0.032910</td>
    </tr>
    <tr>
      <th>OS</th>
      <td>0.173565</td>
      <td>-0.226778</td>
      <td>-0.131021</td>
      <td>0.057994</td>
      <td>-0.341969</td>
      <td>0.084816</td>
      <td>0.010177</td>
      <td>0.245277</td>
      <td>1.000000</td>
      <td>0.038516</td>
      <td>...</td>
      <td>0.296306</td>
      <td>0.134194</td>
      <td>0.185061</td>
      <td>0.317322</td>
      <td>0.369759</td>
      <td>0.171984</td>
      <td>0.369139</td>
      <td>0.195460</td>
      <td>0.026774</td>
      <td>0.150502</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>-0.028782</td>
      <td>0.063504</td>
      <td>0.219743</td>
      <td>0.091088</td>
      <td>0.150921</td>
      <td>-0.038726</td>
      <td>-0.216964</td>
      <td>0.157336</td>
      <td>0.038516</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.114550</td>
      <td>-0.169886</td>
      <td>-0.110763</td>
      <td>-0.139983</td>
      <td>-0.093888</td>
      <td>-0.156847</td>
      <td>-0.111424</td>
      <td>-0.111075</td>
      <td>0.019750</td>
      <td>-0.114083</td>
    </tr>
    <tr>
      <th>CS</th>
      <td>-0.072094</td>
      <td>0.183564</td>
      <td>0.183831</td>
      <td>0.240417</td>
      <td>0.214000</td>
      <td>-0.029923</td>
      <td>-0.391088</td>
      <td>0.308024</td>
      <td>-0.056811</td>
      <td>0.247482</td>
      <td>...</td>
      <td>-0.217335</td>
      <td>-0.158398</td>
      <td>-0.153482</td>
      <td>-0.254417</td>
      <td>-0.160699</td>
      <td>-0.191380</td>
      <td>-0.191237</td>
      <td>-0.230862</td>
      <td>0.008524</td>
      <td>-0.159442</td>
    </tr>
    <tr>
      <th>Imp</th>
      <td>0.119663</td>
      <td>-0.190939</td>
      <td>-0.167492</td>
      <td>-0.132482</td>
      <td>-0.231572</td>
      <td>0.082411</td>
      <td>0.174399</td>
      <td>0.114151</td>
      <td>0.277512</td>
      <td>-0.229690</td>
      <td>...</td>
      <td>0.260864</td>
      <td>0.197701</td>
      <td>0.177665</td>
      <td>0.267579</td>
      <td>0.229205</td>
      <td>0.181524</td>
      <td>0.263684</td>
      <td>0.246299</td>
      <td>0.011178</td>
      <td>0.181019</td>
    </tr>
    <tr>
      <th>SS</th>
      <td>0.165882</td>
      <td>-0.332188</td>
      <td>-0.244277</td>
      <td>-0.131146</td>
      <td>-0.345415</td>
      <td>0.100304</td>
      <td>0.079988</td>
      <td>0.210130</td>
      <td>0.421709</td>
      <td>-0.208061</td>
      <td>...</td>
      <td>0.388186</td>
      <td>0.213684</td>
      <td>0.243609</td>
      <td>0.405578</td>
      <td>0.365536</td>
      <td>0.218884</td>
      <td>0.378285</td>
      <td>0.305635</td>
      <td>0.047986</td>
      <td>0.250599</td>
    </tr>
    <tr>
      <th>Alcohol</th>
      <td>-0.020281</td>
      <td>-0.028773</td>
      <td>-0.002537</td>
      <td>0.130657</td>
      <td>0.076361</td>
      <td>0.053825</td>
      <td>-0.001882</td>
      <td>0.092303</td>
      <td>0.032711</td>
      <td>-0.021806</td>
      <td>...</td>
      <td>0.073585</td>
      <td>-0.033786</td>
      <td>0.065895</td>
      <td>0.027828</td>
      <td>0.012123</td>
      <td>-0.079022</td>
      <td>0.027419</td>
      <td>0.062264</td>
      <td>-0.036110</td>
      <td>0.025353</td>
    </tr>
    <tr>
      <th>Amphet</th>
      <td>0.168624</td>
      <td>-0.246748</td>
      <td>-0.222440</td>
      <td>-0.153895</td>
      <td>-0.395836</td>
      <td>0.084407</td>
      <td>0.131120</td>
      <td>-0.041095</td>
      <td>0.221116</td>
      <td>-0.148868</td>
      <td>...</td>
      <td>0.521537</td>
      <td>0.360896</td>
      <td>0.373587</td>
      <td>0.474993</td>
      <td>0.419616</td>
      <td>0.397671</td>
      <td>0.427977</td>
      <td>0.352249</td>
      <td>0.029641</td>
      <td>0.269259</td>
    </tr>
    <tr>
      <th>Amyl</th>
      <td>-0.029705</td>
      <td>-0.098108</td>
      <td>-0.162250</td>
      <td>-0.002560</td>
      <td>0.080596</td>
      <td>0.038790</td>
      <td>0.033317</td>
      <td>0.030199</td>
      <td>0.060511</td>
      <td>-0.096057</td>
      <td>...</td>
      <td>0.358947</td>
      <td>0.129242</td>
      <td>0.343731</td>
      <td>0.260235</td>
      <td>0.161425</td>
      <td>0.057083</td>
      <td>0.212167</td>
      <td>0.223269</td>
      <td>0.006498</td>
      <td>0.151936</td>
    </tr>
    <tr>
      <th>Benzos</th>
      <td>0.156942</td>
      <td>-0.126416</td>
      <td>-0.133717</td>
      <td>-0.133038</td>
      <td>-0.401114</td>
      <td>0.118517</td>
      <td>0.272221</td>
      <td>-0.103442</td>
      <td>0.201334</td>
      <td>-0.164512</td>
      <td>...</td>
      <td>0.342400</td>
      <td>0.427162</td>
      <td>0.297018</td>
      <td>0.357506</td>
      <td>0.302735</td>
      <td>0.519292</td>
      <td>0.344434</td>
      <td>0.303475</td>
      <td>0.026093</td>
      <td>0.274071</td>
    </tr>
    <tr>
      <th>Caffeine</th>
      <td>-0.006448</td>
      <td>0.037413</td>
      <td>-0.011195</td>
      <td>0.036019</td>
      <td>0.004974</td>
      <td>0.084292</td>
      <td>0.013032</td>
      <td>0.054343</td>
      <td>0.027304</td>
      <td>-0.016190</td>
      <td>...</td>
      <td>0.033983</td>
      <td>0.014683</td>
      <td>0.013997</td>
      <td>0.006747</td>
      <td>-0.004510</td>
      <td>0.023196</td>
      <td>0.040013</td>
      <td>0.126669</td>
      <td>-0.011084</td>
      <td>0.057259</td>
    </tr>
    <tr>
      <th>Cannabis</th>
      <td>0.209274</td>
      <td>-0.446846</td>
      <td>-0.301579</td>
      <td>-0.265405</td>
      <td>-0.551938</td>
      <td>0.136049</td>
      <td>0.095535</td>
      <td>-0.014369</td>
      <td>0.414163</td>
      <td>-0.148481</td>
      <td>...</td>
      <td>0.552978</td>
      <td>0.233546</td>
      <td>0.309144</td>
      <td>0.554048</td>
      <td>0.521048</td>
      <td>0.295021</td>
      <td>0.579934</td>
      <td>0.515145</td>
      <td>0.043969</td>
      <td>0.271175</td>
    </tr>
    <tr>
      <th>Choco</th>
      <td>-0.060475</td>
      <td>0.047522</td>
      <td>0.074584</td>
      <td>0.027837</td>
      <td>0.124505</td>
      <td>0.024475</td>
      <td>0.012583</td>
      <td>0.020305</td>
      <td>0.001239</td>
      <td>0.036266</td>
      <td>...</td>
      <td>-0.049131</td>
      <td>-0.078250</td>
      <td>-0.026126</td>
      <td>-0.055693</td>
      <td>-0.078222</td>
      <td>-0.044426</td>
      <td>-0.073984</td>
      <td>-0.040433</td>
      <td>-0.039510</td>
      <td>-0.073982</td>
    </tr>
    <tr>
      <th>Coke</th>
      <td>0.091650</td>
      <td>-0.220664</td>
      <td>-0.176704</td>
      <td>-0.108433</td>
      <td>-0.277531</td>
      <td>0.086853</td>
      <td>0.139915</td>
      <td>0.030942</td>
      <td>0.188276</td>
      <td>-0.198147</td>
      <td>...</td>
      <td>0.610783</td>
      <td>0.424215</td>
      <td>0.441656</td>
      <td>0.412975</td>
      <td>0.384027</td>
      <td>0.343801</td>
      <td>0.426584</td>
      <td>0.401253</td>
      <td>0.046451</td>
      <td>0.281865</td>
    </tr>
    <tr>
      <th>Crack</th>
      <td>0.082169</td>
      <td>-0.051355</td>
      <td>-0.151324</td>
      <td>-0.147743</td>
      <td>-0.229899</td>
      <td>0.035478</td>
      <td>0.111435</td>
      <td>-0.050969</td>
      <td>0.097002</td>
      <td>-0.103945</td>
      <td>...</td>
      <td>0.232394</td>
      <td>0.527120</td>
      <td>0.229557</td>
      <td>0.194630</td>
      <td>0.230271</td>
      <td>0.355952</td>
      <td>0.261605</td>
      <td>0.241017</td>
      <td>0.026438</td>
      <td>0.250285</td>
    </tr>
    <tr>
      <th>Ecstasy</th>
      <td>0.167231</td>
      <td>-0.384784</td>
      <td>-0.228574</td>
      <td>-0.159819</td>
      <td>-0.336328</td>
      <td>0.071826</td>
      <td>0.069948</td>
      <td>0.078822</td>
      <td>0.296306</td>
      <td>-0.114550</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.265752</td>
      <td>0.508329</td>
      <td>0.554134</td>
      <td>0.570946</td>
      <td>0.259177</td>
      <td>0.547536</td>
      <td>0.380399</td>
      <td>0.039119</td>
      <td>0.251542</td>
    </tr>
    <tr>
      <th>Heroin</th>
      <td>0.091180</td>
      <td>-0.121675</td>
      <td>-0.136728</td>
      <td>-0.131051</td>
      <td>-0.300210</td>
      <td>0.042881</td>
      <td>0.172685</td>
      <td>-0.079998</td>
      <td>0.134194</td>
      <td>-0.169886</td>
      <td>...</td>
      <td>0.265752</td>
      <td>1.000000</td>
      <td>0.249613</td>
      <td>0.242205</td>
      <td>0.279543</td>
      <td>0.478264</td>
      <td>0.268299</td>
      <td>0.224314</td>
      <td>0.016959</td>
      <td>0.256073</td>
    </tr>
    <tr>
      <th>Ketamine</th>
      <td>0.074800</td>
      <td>-0.220689</td>
      <td>-0.189825</td>
      <td>-0.076479</td>
      <td>-0.112577</td>
      <td>0.031961</td>
      <td>0.062750</td>
      <td>0.018727</td>
      <td>0.185061</td>
      <td>-0.110763</td>
      <td>...</td>
      <td>0.508329</td>
      <td>0.249613</td>
      <td>1.000000</td>
      <td>0.410852</td>
      <td>0.409933</td>
      <td>0.186836</td>
      <td>0.410059</td>
      <td>0.255897</td>
      <td>0.059400</td>
      <td>0.196172</td>
    </tr>
    <tr>
      <th>LegalH</th>
      <td>0.220806</td>
      <td>-0.419839</td>
      <td>-0.315676</td>
      <td>-0.208656</td>
      <td>-0.426030</td>
      <td>0.077511</td>
      <td>0.113342</td>
      <td>-0.037383</td>
      <td>0.317322</td>
      <td>-0.139983</td>
      <td>...</td>
      <td>0.554134</td>
      <td>0.242205</td>
      <td>0.410852</td>
      <td>1.000000</td>
      <td>0.471942</td>
      <td>0.324824</td>
      <td>0.531819</td>
      <td>0.341127</td>
      <td>0.015995</td>
      <td>0.319639</td>
    </tr>
    <tr>
      <th>LSD</th>
      <td>0.215234</td>
      <td>-0.323207</td>
      <td>-0.278983</td>
      <td>-0.177817</td>
      <td>-0.498263</td>
      <td>0.129031</td>
      <td>0.037095</td>
      <td>0.018166</td>
      <td>0.369759</td>
      <td>-0.093888</td>
      <td>...</td>
      <td>0.570946</td>
      <td>0.279543</td>
      <td>0.409933</td>
      <td>0.471942</td>
      <td>1.000000</td>
      <td>0.266792</td>
      <td>0.668627</td>
      <td>0.292235</td>
      <td>0.071252</td>
      <td>0.286189</td>
    </tr>
    <tr>
      <th>Meth</th>
      <td>0.175429</td>
      <td>-0.191503</td>
      <td>-0.181489</td>
      <td>-0.170103</td>
      <td>-0.413946</td>
      <td>0.063805</td>
      <td>0.184672</td>
      <td>-0.121708</td>
      <td>0.171984</td>
      <td>-0.156847</td>
      <td>...</td>
      <td>0.259177</td>
      <td>0.478264</td>
      <td>0.186836</td>
      <td>0.324824</td>
      <td>0.266792</td>
      <td>1.000000</td>
      <td>0.277258</td>
      <td>0.220544</td>
      <td>-0.003798</td>
      <td>0.258825</td>
    </tr>
    <tr>
      <th>Shrooms</th>
      <td>0.202910</td>
      <td>-0.331456</td>
      <td>-0.272431</td>
      <td>-0.169762</td>
      <td>-0.490052</td>
      <td>0.115962</td>
      <td>0.042386</td>
      <td>0.021105</td>
      <td>0.369139</td>
      <td>-0.111424</td>
      <td>...</td>
      <td>0.547536</td>
      <td>0.268299</td>
      <td>0.410059</td>
      <td>0.531819</td>
      <td>0.668627</td>
      <td>0.277258</td>
      <td>1.000000</td>
      <td>0.323034</td>
      <td>0.099120</td>
      <td>0.245239</td>
    </tr>
    <tr>
      <th>Nicotine</th>
      <td>0.063197</td>
      <td>-0.248883</td>
      <td>-0.192084</td>
      <td>-0.240547</td>
      <td>-0.277913</td>
      <td>0.077724</td>
      <td>0.128430</td>
      <td>-0.019196</td>
      <td>0.195460</td>
      <td>-0.111075</td>
      <td>...</td>
      <td>0.380399</td>
      <td>0.224314</td>
      <td>0.255897</td>
      <td>0.341127</td>
      <td>0.292235</td>
      <td>0.220544</td>
      <td>0.323034</td>
      <td>1.000000</td>
      <td>0.026760</td>
      <td>0.255023</td>
    </tr>
    <tr>
      <th>Semer</th>
      <td>0.050454</td>
      <td>-0.049729</td>
      <td>0.013354</td>
      <td>-0.036342</td>
      <td>-0.068018</td>
      <td>0.022716</td>
      <td>-0.001673</td>
      <td>0.022909</td>
      <td>0.026774</td>
      <td>0.019750</td>
      <td>...</td>
      <td>0.039119</td>
      <td>0.016959</td>
      <td>0.059400</td>
      <td>0.015995</td>
      <td>0.071252</td>
      <td>-0.003798</td>
      <td>0.099120</td>
      <td>0.026760</td>
      <td>1.000000</td>
      <td>0.056072</td>
    </tr>
    <tr>
      <th>VSA</th>
      <td>0.101165</td>
      <td>-0.229657</td>
      <td>-0.134852</td>
      <td>-0.120540</td>
      <td>-0.267033</td>
      <td>0.087011</td>
      <td>0.115086</td>
      <td>-0.032910</td>
      <td>0.150502</td>
      <td>-0.114083</td>
      <td>...</td>
      <td>0.251542</td>
      <td>0.256073</td>
      <td>0.196172</td>
      <td>0.319639</td>
      <td>0.286189</td>
      <td>0.258825</td>
      <td>0.245239</td>
      <td>0.255023</td>
      <td>0.056072</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>32 rows × 32 columns</p>
</div>




```python
# Plotting Heatmaps for Correlations between all the features
plt.figure(figsize=(30,30))
sns.heatmap(drug_corr, cbar = True,  square = True, annot=True, fmt= '.2f', annot_kws={'size': 12}, cmap= 'coolwarm')
plt.title('Correlation between all the features')
```




    Text(0.5, 1.0, 'Correlation between all the features')




    
![png](Drug_Consumption_files/Drug_Consumption_23_1.png)
    


From the above heatmap, it is a bit of a pain to manually go through each feature and pick out which ones are significiant given that there are 31 features (excluding the ID and Target Column). To avoid manually doing this, we will use the sci-kit learn's automatic feature selection to simplify things. 

### Automated Feature Selection


```python
X = df[column_names]
y = df['Drug Use']
```


```python
drug_mutual_information = mutual_info_classif(X, y)

plt.subplots(1, figsize=(26, 1))
sns.heatmap(drug_mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
plt.yticks([], [])
plt.gca().set_xticklabels(X.columns, rotation=45, ha='right', fontsize=12)
plt.suptitle("Variable Importance (mutual_info_classif)", fontsize=18, y=1.2)
plt.gcf().subplots_adjust(wspace=0.2)
```


    
![png](Drug_Consumption_files/Drug_Consumption_27_0.png)
    


From the above plot, we see the mutual variable importance of each feature. Mutual information (MI) is a non-negative value measuring the dependency between two variables. If two varaibles are independent of each other then, the MI score will be 0. The higher the MI score, the more dependent the target outcome is on that feature. The scores are determined using methods based on entropy estimation from KNN distances. 
 Based on the MI scores above, Cannabis is the most important feature. Now in order to minimize the number of features to the features we are deeming influential on determining the target outcome, we will initially take the top 50% of the features based on there.


```python
trans = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=50)
kepler_X_trans = trans.fit_transform(X, y)
print("We started with {0} features but retained only {1} of them!".format(X.shape[1] - 1, kepler_X_trans.shape[1]))
```

    We started with 31 features but retained only 16 of them!



```python
columns_retained_Select = X.columns[trans.get_support()].values
X=pd.DataFrame(kepler_X_trans, columns=columns_retained_Select)
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Age</th>
      <th>Cntry</th>
      <th>SS</th>
      <th>Amphet</th>
      <th>Benzos</th>
      <th>Cannabis</th>
      <th>Coke</th>
      <th>Ecstasy</th>
      <th>Ketamine</th>
      <th>LegalH</th>
      <th>LSD</th>
      <th>Meth</th>
      <th>Shrooms</th>
      <th>Nicotine</th>
      <th>VSA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.49788</td>
      <td>0.96082</td>
      <td>-1.18084</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>-0.07854</td>
      <td>0.96082</td>
      <td>-0.21575</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.49788</td>
      <td>0.96082</td>
      <td>0.40148</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>-0.95197</td>
      <td>0.96082</td>
      <td>-1.18084</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.49788</td>
      <td>0.96082</td>
      <td>-0.21575</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>1884.0</td>
      <td>-0.95197</td>
      <td>-0.57009</td>
      <td>1.92173</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>1885.0</td>
      <td>-0.95197</td>
      <td>-0.57009</td>
      <td>0.76540</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1882</th>
      <td>1886.0</td>
      <td>-0.07854</td>
      <td>-0.57009</td>
      <td>-0.52593</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1883</th>
      <td>1887.0</td>
      <td>-0.95197</td>
      <td>-0.57009</td>
      <td>1.22470</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1884</th>
      <td>1888.0</td>
      <td>-0.95197</td>
      <td>0.21128</td>
      <td>1.22470</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>1885 rows × 16 columns</p>
</div>




```python
y = df['Drug Use']
```

## Oversampling

In order to address the imbalance between the majority class (More Likely to Use Drugs) and the minority class (Not Likely to use Drugs), we will employ the method of oversampling to prevent the model from overfitting due to the large number of the majority class.  


```python
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
```

    (1319, 16)
    (1319,)
    (566, 16)
    (566,)



```python
print("Before oversampling: ", Counter(y_train))
```

    Before oversampling:  Counter({1: 1000, 0: 319})



```python
oversample = RandomOverSampler(sampling_strategy='minority')
```


```python
# fit and apply the transform
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
```


```python
# summarize class distribution
print("After undersampling: ", Counter(y_train_over))
```

    After undersampling:  Counter({1: 1000, 0: 1000})


We increased the number of minority class from 319 to 1000 so the ratio bewteen the majority to minority is 1:1. 

## Undersampling

In order to address the imbalance between the majority class (More Likely to Use Drugs) and the minority class (Not Likely to use Drugs), we will employ the method of undersampling to prevent the model from overfitting due to the large number of the majority class.  


```python
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
```

    (1319, 16)
    (1319,)
    (566, 16)
    (566,)



```python
print("Before undersampling: ", Counter(y_train))
```

    Before undersampling:  Counter({1: 1000, 0: 319})



```python
# define undersampling strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
```


```python
# fit and apply the transform
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
```


```python
# summarize class distribution
print("After undersampling: ", Counter(y_train_under))
```

    After undersampling:  Counter({0: 319, 1: 319})


We reduced the number of majority class from 1000 to 319 so the ratio bewteen the majority to minority is 1:1. 

# Modeling

We will be using GridSearchCV to find the best model to use for our data. Moreover, it allows us to implement cross validation to prevent us from using a single train test split later on. 


```python
# Creating a function to calculate best model for this problem
def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10,20]
            }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        'guassian bayes naive ': {
            'model': GaussianNB(),
            'parameters': {
               }
        },
        'LDA': {
            'model': LinearDiscriminantAnalysis(),
            'parameters': {
               }
        },
        'LASSO': {
            'model': linear_model.Lasso(alpha=0.1),
            'parameters': {
               }
        },
        'QDA': {
            'model': QDA(),
            'parameters': {
               }
        },
        'RIDGE': {
            'model': Ridge(alpha=1.0),
            'parameters': {
               }
        },


    }
    
    scores = [] 
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score']).sort_values(by='score', ascending=False)


```

##  GridSearchCV Without Undersampling


```python
find_best_model(X_train, y_train)
```

    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>best_parameters</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>decision_tree</td>
      <td>{'criterion': 'gini', 'max_depth': 5}</td>
      <td>0.995449</td>
    </tr>
    <tr>
      <th>2</th>
      <td>random_forest</td>
      <td>{'n_estimators': 50}</td>
      <td>0.993176</td>
    </tr>
    <tr>
      <th>0</th>
      <td>logistic_regression</td>
      <td>{'C': 10}</td>
      <td>0.971189</td>
    </tr>
    <tr>
      <th>6</th>
      <td>QDA</td>
      <td>{}</td>
      <td>0.937055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LDA</td>
      <td>{}</td>
      <td>0.930997</td>
    </tr>
    <tr>
      <th>3</th>
      <td>guassian bayes naive</td>
      <td>{}</td>
      <td>0.914302</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RIDGE</td>
      <td>{}</td>
      <td>0.490626</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LASSO</td>
      <td>{}</td>
      <td>0.464053</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, we see that without undersampling, most of our scores are very high, nearly reaching 1.0 which is a clear sign that we are overfitting. So let's not check what happens with the scores with the undersampled data. 

## GridSearch CV With Undersampling 


```python
find_best_model(X_train_under, y_train_under)
```

    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>best_parameters</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>decision_tree</td>
      <td>{'criterion': 'gini', 'max_depth': 5}</td>
      <td>0.996863</td>
    </tr>
    <tr>
      <th>2</th>
      <td>random_forest</td>
      <td>{'n_estimators': 20}</td>
      <td>0.993725</td>
    </tr>
    <tr>
      <th>0</th>
      <td>logistic_regression</td>
      <td>{'C': 10}</td>
      <td>0.976501</td>
    </tr>
    <tr>
      <th>6</th>
      <td>QDA</td>
      <td>{}</td>
      <td>0.945140</td>
    </tr>
    <tr>
      <th>3</th>
      <td>guassian bayes naive</td>
      <td>{}</td>
      <td>0.938866</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LDA</td>
      <td>{}</td>
      <td>0.895017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RIDGE</td>
      <td>{}</td>
      <td>0.128602</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LASSO</td>
      <td>{}</td>
      <td>0.121150</td>
    </tr>
  </tbody>
</table>
</div>



With the undersampled train data, we see that the scores virtually stay the same, but we can say that these scores are more accurate as undersampling prevents us from overfitting in many cases. In this specific case when choosing a model, I would not choose the models that have a score of 0.99 as those are too high and more prone to be overfitted. As a result it would be best to choose a model like Quadratic Discrimenant Analysis (QDA) model or the Logisitic Regression Model as they both have high scores but are not exactly 1.0. 

## GridSearch CV With Oversampling


```python
find_best_model(X_train_over, y_train_over)
```

    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>best_parameters</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>decision_tree</td>
      <td>{'criterion': 'gini', 'max_depth': 10}</td>
      <td>0.997500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>random_forest</td>
      <td>{'n_estimators': 15}</td>
      <td>0.997500</td>
    </tr>
    <tr>
      <th>0</th>
      <td>logistic_regression</td>
      <td>{'C': 5}</td>
      <td>0.974000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>QDA</td>
      <td>{}</td>
      <td>0.936500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>guassian bayes naive</td>
      <td>{}</td>
      <td>0.918000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LDA</td>
      <td>{}</td>
      <td>0.897000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RIDGE</td>
      <td>{}</td>
      <td>0.344453</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LASSO</td>
      <td>{}</td>
      <td>0.325839</td>
    </tr>
  </tbody>
</table>
</div>




```python
scores = cross_val_score(QDA(), X_train_under, y_train_under, cv=5)
print('Average Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))
```

    Average Accuracy : 95%



```python
scores = cross_val_score(QDA(), X_train_over, y_train_over, cv=5)
print('Average Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))
```

    Average Accuracy : 94%



```python
scores = cross_val_score(LogisticRegression(solver='lbfgs', multi_class='auto',C=10), X_train_under, y_train_under, cv=5)
print('Average Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))
```

    Average Accuracy : 98%


    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
scores = cross_val_score(LogisticRegression(solver='lbfgs', multi_class='auto',C=5), X_train_over, y_train_over, cv=5)
print('Average Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))
```

    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\ramay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


    Average Accuracy : 97%


Based on the average accuracy scores of each model, the Logistic Regression Model had the best average accuracy regardless of sampling method. Now we must choose which sampling method returrns the best model. 

## Logistic Regression

### Undersampling Logistic Regression Model


```python
# Create a pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('logistic', LogisticRegression(solver='lbfgs', multi_class='auto',C=10))])

pipe.fit(X_train_under, y_train_under)
# Get Model Performance
print('Logistic Regression Training Score: \n',pipe.score(X_train_under, y_train_under))
print('Logistic Regression Test Score: \n', pipe.score(X_test, y_test))
print('Classification Report: \n', classification_report(y_test,pipe.predict(X_test)))
score_logreg_train = pipe.score(X_train_under, y_train_under)
score_logreg_test = pipe.score(X_test, y_test)
cm_logreg = metrics.confusion_matrix(y_test, pipe.predict(X_test))
modified_cm = []
for index,value in enumerate(cm_logreg):
    if index == 0:
        modified_cm.append(['TN = ' + str(value[0]), 'FP = ' + str(value[1])])
    if index == 1:
        modified_cm.append(['FN = ' + str(value[0]), 'TP = ' + str(value[1])])   
        
plt.figure(figsize=(9,9))
p=sns.heatmap(cm_logreg, annot=np.array(modified_cm),
            fmt="",
            annot_kws={"size": 20},
            linewidths=.5,
            square = True,
            cmap = 'Blues',
            xticklabels = [ 'Not Likely to Use','Likely to Use'],
            yticklabels = ['Not Likely to Use','Likely to Use'],
            )

plt.ylabel('Actual label', fontsize = 17)
plt.xlabel('Predicted label', fontsize = 17)
plt.title('Accuracy score post logistic regression: {:.3f}'.format(score_logreg_test), size = 17)
plt.tick_params(labelsize= 15)
plt.savefig('logreg_cm.png')
```

    Logistic Regression Training Score: 
     0.9749216300940439
    Logistic Regression Test Score: 
     0.9575971731448764
    Classification Report: 
                   precision    recall  f1-score   support
    
               0       0.87      0.98      0.92       148
               1       0.99      0.95      0.97       418
    
        accuracy                           0.96       566
       macro avg       0.93      0.96      0.95       566
    weighted avg       0.96      0.96      0.96       566
    



    
![png](Drug_Consumption_files/Drug_Consumption_66_1.png)
    


### Oversampling Logistic Regression Model


```python
# Create a pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('logistic', LogisticRegression(solver='lbfgs', multi_class='auto',C=5))])

pipe.fit(X_train_over, y_train_over)
# Get Model Performance
print('Logistic Regression Training Score: \n',pipe.score(X_train_over, y_train_over))
print('Logistic Regression Test Score: \n', pipe.score(X_test, y_test))
print('Classification Report: \n', classification_report(y_test,pipe.predict(X_test)))
score_logreg_train = pipe.score(X_train_over, y_train_over)
score_logreg_test = pipe.score(X_test, y_test)
cm_logreg = metrics.confusion_matrix(y_test, pipe.predict(X_test))
modified_cm = []
for index,value in enumerate(cm_logreg):
    if index == 0:
        modified_cm.append(['TN = ' + str(value[0]), 'FP = ' + str(value[1])])
    if index == 1:
        modified_cm.append(['FN = ' + str(value[0]), 'TP = ' + str(value[1])])   
        
plt.figure(figsize=(9,9))
p=sns.heatmap(cm_logreg, annot=np.array(modified_cm),
            fmt="",
            annot_kws={"size": 20},
            linewidths=.5,
            square = True,
            cmap = 'Blues',
            xticklabels = [ 'Not Likely to Use','Likely to Use'],
            yticklabels = [ 'Not Likely to Use','Likely to Use'],
            )

plt.ylabel('Actual label', fontsize = 17)
plt.xlabel('Predicted label', fontsize = 17)
plt.title('Accuracy score post logistic regression: {:.3f}'.format(score_logreg_test), size = 17)
plt.tick_params(labelsize= 15)
plt.savefig('logreg_cm.png')
```

    Logistic Regression Training Score: 
     0.9745
    Logistic Regression Test Score: 
     0.9558303886925795
    Classification Report: 
                   precision    recall  f1-score   support
    
               0       0.87      0.98      0.92       148
               1       0.99      0.95      0.97       418
    
        accuracy                           0.96       566
       macro avg       0.93      0.96      0.95       566
    weighted avg       0.96      0.96      0.96       566
    



    
![png](Drug_Consumption_files/Drug_Consumption_68_1.png)
    


From the individual model performance, it would be best to go with the  Oversampling Logisitc Regression Model as it seems that this specific Logistic Regression Model has a Training Score which is higher than its test score. For accuracy scores, we typically want the test score to be less than the training score, but it is't a requirements by any means. Moreover, it is a more generalizable model than the undersampled model. 
