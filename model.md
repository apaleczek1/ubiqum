## Table of contents:
* [The main purpose](#first-bullet)
* [Importing necessary modules](#second-bullet)
* [Exploring data](#third-bullet)
* [Features extraction](#featuresextraction)
* [Data visualization](#datavisualization)
* [Modeling](#modeling)
* [Conclusions](#conclusions)

## Questions:
* [Do customers in different regions spend more per transaction? Which regions spend the most/least?](#Q1)
* [Is there a relationship between the number of items purchased and amount spent?](#Q2)
* [Are there differences in the age of customers between regions?](#Q3)
* [If so, can we predict the age of a customer in a region based on other demographic data?](#Q4)
* [Is there any correlation between age of a customer and if the transaction was made online or in the store? Do any other factors predict if a customer will buy online or in our stores?](#Q5)
* [What about items? Are these even in the tree? Why or why not??](#Q6)

# The main purpose <a class="anchor" id="first-bullet"></a>

The main purpose of these analyzes is to understand and discover the behavior of customers in online and in-store shopping from different regions. One of the tasks set by the client was age prediction based on the region. Another dependence that played a large role for the customer and understanding the behavior of buyers was the prediction whether the customer will buy the product online or in the store based on the collected demographic data.

# Importing necessary modules <a class="anchor" id="second-bullet"></a>


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline


#Machine learning packages
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
from sklearn import tree

import xgboost as xgb


```

# Exploring data <a class="anchor" id="third-bullet"></a>


```python
data = pd.read_csv('Demographic_Data.csv')
```

#### The dataset contains 80,000 records and five columns:
#### a) in-store - the bool value determines whether the purchase was made online or in-store 
#### b) age - the age of the client (min 18 max 85)
#### c) items - number of items purchased (min 1 max 8)
#### d) amout - amount paid (min 5.0047 max 3000) 
#### e) region - a categorical, nominal variable specifies one of 4 regions (North, South, East, West) 


```python
data = data.drop_duplicates()
```

### 21 duplicates were removed from the dataset. I decided to remove all duplicates, because the amount of duplicates were not high and indexes of transactions were different (not in sequence). Dataset do not contain missing values.

# Features extraction <a class="anchor" id="featuresextraction"></a>

Age rounded to nearest tenth


```python
data['age_round'] = ((data['age']/10).round()*10).astype(int)
```

Amount spend per item


```python
data['amount_per_item'] = data['amount']/data['items']
```

Counting people in each age group (every 10 years).


```python
age_counts = data['age_round'].value_counts().sort_index()
```

Features discretization


```python
data['age_levels']=pd.cut(data['age'],3,labels=['Young','Middle','Old'])
print('Mean of ages in groups:\n', data['age'].groupby(data['age_levels']).mean())
print('Minimum of ages in groups:\n', data['age'].groupby(data['age_levels']).min())
print('Maximum of ages in groups:\n', data['age'].groupby(data['age_levels']).max())
print('Count of ages in groups:\n', data['age'].groupby(data['age_levels']).count())
young_mean = round(data['age'].groupby(data['age_levels']).mean()['Young'])
middle_mean = round(data['age'].groupby(data['age_levels']).mean()['Middle'])
old_mean = round(data['age'].groupby(data['age_levels']).mean()['Old'])

data.loc[data['age_levels']=='Young', 'age_levels_mean'] = young_mean
data.loc[data['age_levels']=='Middle', 'age_levels_mean'] = middle_mean
data.loc[data['age_levels']=='Old', 'age_levels_mean'] = old_mean
```

    Mean of ages in groups:
     age_levels
    Young     30.696913
    Middle    50.602622
    Old       71.654716
    Name: age, dtype: float64
    Minimum of ages in groups:
     age_levels
    Young     18
    Middle    41
    Old       63
    Name: age, dtype: int64
    Maximum of ages in groups:
     age_levels
    Young     40
    Middle    62
    Old       85
    Name: age, dtype: int64
    Count of ages in groups:
     age_levels
    Young     32885
    Middle    34403
    Old       12691
    Name: age, dtype: int64
    


```python
data['age_levels_5']=pd.cut(data['age'],5,labels=['Age1', 'Age2', 'Age3', 'Age4', 'Age5'])
data.loc[data['age_levels_5']=='Age1', 'age_levels_5_mean'] = round(data['age'].groupby(data['age_levels_5']).mean()['Age1'])
data.loc[data['age_levels_5']=='Age2', 'age_levels_5_mean'] = round(data['age'].groupby(data['age_levels_5']).mean()['Age2'])
data.loc[data['age_levels_5']=='Age3', 'age_levels_5_mean'] = round(data['age'].groupby(data['age_levels_5']).mean()['Age3'])
data.loc[data['age_levels_5']=='Age4', 'age_levels_5_mean'] = round(data['age'].groupby(data['age_levels_5']).mean()['Age4'])
data.loc[data['age_levels_5']=='Age5', 'age_levels_5_mean'] = round(data['age'].groupby(data['age_levels_5']).mean()['Age5'])

data['amount_levels_5']=pd.cut(data['amount'],5,labels=['Amount1', 'Amount2', 'Amount3', 'Amount4', 'Amount5'])
data.loc[data['amount_levels_5']=='Amount1', 'amount_levels_5_mean'] = round(data['amount'].groupby(data['amount_levels_5']).mean()['Amount1'])
data.loc[data['amount_levels_5']=='Amount2', 'amount_levels_5_mean'] = round(data['amount'].groupby(data['amount_levels_5']).mean()['Amount2'])
data.loc[data['amount_levels_5']=='Amount3', 'amount_levels_5_mean'] = round(data['amount'].groupby(data['amount_levels_5']).mean()['Amount3'])
data.loc[data['amount_levels_5']=='Amount4', 'amount_levels_5_mean'] = round(data['amount'].groupby(data['amount_levels_5']).mean()['Amount4'])
data.loc[data['amount_levels_5']=='Amount5', 'amount_levels_5_mean'] = round(data['amount'].groupby(data['amount_levels_5']).mean()['Amount5'])

data['amount_normalized']=(data['amount']-data['amount'].mean())/data['amount'].std()

data1 = pd.get_dummies(data['region'], prefix='Region', drop_first=True)
data = pd.concat([data, data1], axis=1)
```

# Data visualization  <a class="anchor" id="datavisualization"></a>

## Simple visualization of data <a class="anchor" id="fifth-a-bullet"></a>


```python
labels = ["in-store purchases", "online purchases"]
plt.pie([data[data['in-store']==1]['in-store'].count(), data[data['in-store']==0]['in-store'].count()], labels=labels, autopct='%1.0f%%')
plt.show()
```


    
![png](output_22_0.png)
    



```python
labels = ["in-store purchases", "online purchases"]
plt.figure(figsize=(12,8))
plt.subplot(2, 2, 1)
plt.pie([data[(data['in-store']==1)&(data['region']==1)]['in-store'].count(), data[(data['in-store']==0)&(data['region']==1)]['in-store'].count()], labels=labels, autopct='%1.0f%%')
plt.title("North")
plt.subplot(2, 2, 2)
plt.pie([data[(data['in-store']==1)&(data['region']==2)]['in-store'].count(), data[(data['in-store']==0)&(data['region']==2)]['in-store'].count()], labels=labels, autopct='%1.0f%%')
plt.title("South")
plt.subplot(2, 2, 3)
plt.pie([data[(data['in-store']==1)&(data['region']==3)]['in-store'].count(), data[(data['in-store']==0)&(data['region']==3)]['in-store'].count()], labels=labels, autopct='%1.0f%%')
plt.title("East")
plt.subplot(2, 2, 4)
plt.pie([data[(data['in-store']==1)&(data['region']==4)]['in-store'].count(), data[(data['in-store']==0)&(data['region']==4)]['in-store'].count()], labels=labels, autopct='%1.0f%%')
plt.title("West")
plt.show()
```


    
![png](output_23_0.png)
    


### North customers buy only in-store, South customers buy only online. More than 60% of East customers buy instore. Half of West customers buy online.


```python
labels = ["North", "South", "East", "West"]

plt.pie([data[data['region']==1]['region'].count(), data[data['region']==2]['region'].count(), data[data['region']==3]['region'].count(), data[data['region']==4]['region'].count()], labels=labels, autopct='%1.0f%%')
plt.show()
```


    
![png](output_25_0.png)
    



```python
plt.hist(data['age'])
plt.xlabel('age')
plt.ylabel('count')
plt.show()
```


    
![png](output_26_0.png)
    


### Age data histogram is Positively (Right) Skewed. 


```python
plt.hist(data['amount'])
plt.xlabel('amount')
plt.ylabel('count')
plt.show()
```


    
![png](output_28_0.png)
    


### Most often, less than 750 were spend when shopping.

# Do customers in different regions spend more per transaction? Which regions spend the most/least? <a class="anchor" id="Q1"></a>


```python
labels = ["North", "South", "East", "West"]
plt.bar(labels,data['amount'].groupby(data['region']).sum()/data['region'].groupby(data['region']).count())
plt.show()
```


    
![png](output_31_0.png)
    


### The bar plot shows that customers from West spend the most per transaction and customers from South spend the least per transaction

# Is there a relationship between the number of items purchased and amount spent?  <a class="anchor" id="Q2"></a>


```python
data.corr('pearson')
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
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
      <th>age_round</th>
      <th>amount_per_item</th>
      <th>age_levels_5_mean</th>
      <th>amount_levels_5_mean</th>
      <th>amount_normalized</th>
      <th>Region_2</th>
      <th>Region_3</th>
      <th>Region_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>in-store</th>
      <td>1.000000</td>
      <td>-0.178180</td>
      <td>-0.003897</td>
      <td>-0.085573</td>
      <td>-0.133171</td>
      <td>-0.180490</td>
      <td>-0.054597</td>
      <td>-0.177309</td>
      <td>-0.093203</td>
      <td>-0.085573</td>
      <td>-0.577343</td>
      <td>0.119691</td>
      <td>-0.000009</td>
    </tr>
    <tr>
      <th>age</th>
      <td>-0.178180</td>
      <td>1.000000</td>
      <td>0.000657</td>
      <td>-0.282033</td>
      <td>-0.235370</td>
      <td>0.983693</td>
      <td>-0.187250</td>
      <td>0.970109</td>
      <td>-0.273884</td>
      <td>-0.282033</td>
      <td>0.398625</td>
      <td>-0.003826</td>
      <td>-0.309304</td>
    </tr>
    <tr>
      <th>items</th>
      <td>-0.003897</td>
      <td>0.000657</td>
      <td>1.000000</td>
      <td>0.000384</td>
      <td>-0.001904</td>
      <td>0.000210</td>
      <td>-0.469112</td>
      <td>0.000801</td>
      <td>0.000434</td>
      <td>0.000384</td>
      <td>0.002149</td>
      <td>-0.002911</td>
      <td>-0.000458</td>
    </tr>
    <tr>
      <th>amount</th>
      <td>-0.085573</td>
      <td>-0.282033</td>
      <td>0.000384</td>
      <td>1.000000</td>
      <td>0.403486</td>
      <td>-0.276769</td>
      <td>0.666868</td>
      <td>-0.267094</td>
      <td>0.974794</td>
      <td>1.000000</td>
      <td>-0.467248</td>
      <td>0.061376</td>
      <td>0.431044</td>
    </tr>
    <tr>
      <th>region</th>
      <td>-0.133171</td>
      <td>-0.235370</td>
      <td>-0.001904</td>
      <td>0.403486</td>
      <td>1.000000</td>
      <td>-0.231827</td>
      <td>0.269473</td>
      <td>-0.222032</td>
      <td>0.393620</td>
      <td>0.403486</td>
      <td>-0.345855</td>
      <td>0.155499</td>
      <td>0.815993</td>
    </tr>
    <tr>
      <th>age_round</th>
      <td>-0.180490</td>
      <td>0.983693</td>
      <td>0.000210</td>
      <td>-0.276769</td>
      <td>-0.231827</td>
      <td>1.000000</td>
      <td>-0.183685</td>
      <td>0.952719</td>
      <td>-0.268681</td>
      <td>-0.276769</td>
      <td>0.393874</td>
      <td>-0.001191</td>
      <td>-0.306566</td>
    </tr>
    <tr>
      <th>amount_per_item</th>
      <td>-0.054597</td>
      <td>-0.187250</td>
      <td>-0.469112</td>
      <td>0.666868</td>
      <td>0.269473</td>
      <td>-0.183685</td>
      <td>1.000000</td>
      <td>-0.178386</td>
      <td>0.649674</td>
      <td>0.666868</td>
      <td>-0.311520</td>
      <td>0.040389</td>
      <td>0.288070</td>
    </tr>
    <tr>
      <th>age_levels_5_mean</th>
      <td>-0.177309</td>
      <td>0.970109</td>
      <td>0.000801</td>
      <td>-0.267094</td>
      <td>-0.222032</td>
      <td>0.952719</td>
      <td>-0.178386</td>
      <td>1.000000</td>
      <td>-0.259223</td>
      <td>-0.267094</td>
      <td>0.383516</td>
      <td>-0.004165</td>
      <td>-0.293751</td>
    </tr>
    <tr>
      <th>amount_levels_5_mean</th>
      <td>-0.093203</td>
      <td>-0.273884</td>
      <td>0.000434</td>
      <td>0.974794</td>
      <td>0.393620</td>
      <td>-0.268681</td>
      <td>0.649674</td>
      <td>-0.259223</td>
      <td>1.000000</td>
      <td>0.974794</td>
      <td>-0.446413</td>
      <td>0.055100</td>
      <td>0.420443</td>
    </tr>
    <tr>
      <th>amount_normalized</th>
      <td>-0.085573</td>
      <td>-0.282033</td>
      <td>0.000384</td>
      <td>1.000000</td>
      <td>0.403486</td>
      <td>-0.276769</td>
      <td>0.666868</td>
      <td>-0.267094</td>
      <td>0.974794</td>
      <td>1.000000</td>
      <td>-0.467248</td>
      <td>0.061376</td>
      <td>0.431044</td>
    </tr>
    <tr>
      <th>Region_2</th>
      <td>-0.577343</td>
      <td>0.398625</td>
      <td>0.002149</td>
      <td>-0.467248</td>
      <td>-0.345855</td>
      <td>0.393874</td>
      <td>-0.311520</td>
      <td>0.383516</td>
      <td>-0.446413</td>
      <td>-0.467248</td>
      <td>1.000000</td>
      <td>-0.311130</td>
      <td>-0.400548</td>
    </tr>
    <tr>
      <th>Region_3</th>
      <td>0.119691</td>
      <td>-0.003826</td>
      <td>-0.002911</td>
      <td>0.061376</td>
      <td>0.155499</td>
      <td>-0.001191</td>
      <td>0.040389</td>
      <td>-0.004165</td>
      <td>0.055100</td>
      <td>0.061376</td>
      <td>-0.311130</td>
      <td>1.000000</td>
      <td>-0.373886</td>
    </tr>
    <tr>
      <th>Region_4</th>
      <td>-0.000009</td>
      <td>-0.309304</td>
      <td>-0.000458</td>
      <td>0.431044</td>
      <td>0.815993</td>
      <td>-0.306566</td>
      <td>0.288070</td>
      <td>-0.293751</td>
      <td>0.420443</td>
      <td>0.431044</td>
      <td>-0.400548</td>
      <td>-0.373886</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### There are no Pearson correlation between items and amount, but there are observed correlation (not very strong) between age - in-store, region - items, region - age. There are no Spearman correlation between items and amount, but there are observed correlation (not very strong) between age - in-store, region - items, region - age.

## Checking dependency between amount and items purchased online and in-store


```python
plt.bar(['online', 'in store'],data['items'].groupby(data['in-store']).sum())
plt.ylabel('Sum of items buyed')
plt.show()
```


    
![png](output_37_0.png)
    



```python
plt.bar(['online', 'in_store'],data['amount'].groupby(data['in-store']).sum())
plt.ylabel('Spent amount')
plt.show()
```


    
![png](output_38_0.png)
    



```python
plt.bar(['online', 'in-store'],data['amount_per_item'].groupby(data['in-store']).sum())
plt.ylabel('Amount per item spent')
plt.show()
```


    
![png](output_39_0.png)
    


### Total number of items purchased online is same as total number of items purchased online, but customers spend more money during online shoping, because they choose more expensive products than in in-store shopping.

## Checking dependency between age and items, amount and amount per item


```python
plt.bar(data['amount'].groupby(data['age_round']).sum().index, data['amount'].groupby(data['age_round']).sum().values)
plt.xlabel('Age')
plt.ylabel('Sum of amount spent')
plt.show()
```


    
![png](output_42_0.png)
    


### The bar plots shows that distribution beetween three selected dependencies (sum of amount spent, sum of item bought, sum of amount per item) does not change. Due to the unequal size of age groups, further analyzes are needed, especially taking into account the value per person.


```python
plt.bar(data['items'].groupby(data['age_round']).sum().index, data['items'].groupby(data['age_round']).sum().values/age_counts.values)
plt.xlabel('Age')
plt.ylabel('Items per person bought')
plt.show()
```


    
![png](output_44_0.png)
    


### In each age group, customers buy on average the same number of products per person.


```python
plt.bar(data['amount'].groupby(data['age_round']).sum().index, data['amount'].groupby(data['age_round']).sum().values/age_counts.values)
plt.xlabel('Age')
plt.ylabel('Amount per person spend')
plt.show()
```


    
![png](output_46_0.png)
    


### The bar plot shows that younger people spend in average more money


```python
plt.bar(data['amount'].groupby(data['age_round']).sum().index, data['amount'].groupby(data['age_round']).sum().values/data['items'].groupby(data['age_round']).sum().values/age_counts.values)
plt.xlabel('Age')
plt.ylabel('Amount per item per person spend')
plt.show()
```


    
![png](output_48_0.png)
    


### The bar plot shows that younger people spend in average more money per item.


```python
instore = data['amount'][data['in-store']==True].groupby(data['age_round']).sum().index, data['amount'][data['in-store']==True].groupby(data['age_round']).sum().values/data['items'][data['in-store']==True].groupby(data['age_round']).sum().values/age_counts.values[:-1]
online = data['amount'][data['in-store']==False].groupby(data['age_round']).sum().index, data['amount'][data['in-store']==False].groupby(data['age_round']).sum().values/data['items'][data['in-store']==False].groupby(data['age_round']).sum().values/age_counts.values
labels = age_counts.index
x_instore = np.arange(len(labels[:-1]))
x_online = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x_instore - width/2, instore[1], width, label='In-store')
rects2 = ax.bar(x_online + width/2, online[1], width, label='Online')

ax.set_ylabel('Amount per item per person spend')
ax.set_xlabel('Age')
ax.set_xticks(x_online)
ax.set_xticklabels(labels.astype(int))
ax.legend()

fig.tight_layout()

plt.show()
```


    
![png](output_50_0.png)
    


### Presented graph shows that:
### a) People older than 80 years old do not buy in-store
### b) Only people in their seventies bought more expensive products in-store than online
### c) People from 30-60 years old spend more less the same in-store and online per one product
### d) Youngest people spend more per item online than in-store

## Comparision of amount spend in-store and onlne by East and West customers (North and South customers are skipped because they choose only one method of buying)


```python
instore_reg3 = data['amount'][(data['in-store']==True)&(data['region']==3)].groupby(data['age_round']).sum().index, data['amount'][(data['in-store']==True)&(data['region']==3)].groupby(data['age_round']).sum().values/data['items'][(data['in-store']==True)&(data['region']==3)].groupby(data['age_round']).sum().values/age_counts.values[:-1]
online_reg3 = data['amount'][(data['in-store']==False)&(data['region']==3)].groupby(data['age_round']).sum().index, data['amount'][(data['in-store']==False)&(data['region']==3)].groupby(data['age_round']).sum().values/data['items'][(data['in-store']==False)&(data['region']==3)].groupby(data['age_round']).sum().values/age_counts.values[:-2]
labels = age_counts.index
x_instore = np.arange(len(labels[:-1]))
x_online = np.arange(len(labels[:-2]))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x_instore - width/2, instore_reg3[1], width, label='In-store East')
rects2 = ax.bar(x_online + width/2, online_reg3[1], width, label='Online East')

ax.set_ylabel('Amount per item per person spend')
ax.set_xlabel('Age')
ax.set_xticks(x_instore)
ax.set_xticklabels(labels[:-1].astype(int))
ax.legend()

fig.tight_layout()

plt.show()
```


    
![png](output_53_0.png)
    


### People from East spend in average more amount per item online besides 70 yo people which do not buy online


```python
instore_reg3 = data['amount'][(data['in-store']==True)&(data['region']==4)].groupby(data['age_round']).sum().index, data['amount'][(data['in-store']==True)&(data['region']==4)].groupby(data['age_round']).sum().values/data['items'][(data['in-store']==True)&(data['region']==4)].groupby(data['age_round']).sum().values/age_counts.values[:-2]
online_reg3 = data['amount'][(data['in-store']==False)&(data['region']==4)].groupby(data['age_round']).sum().index, data['amount'][(data['in-store']==False)&(data['region']==4)].groupby(data['age_round']).sum().values/data['items'][(data['in-store']==False)&(data['region']==4)].groupby(data['age_round']).sum().values/age_counts.values[:-2]
labels = age_counts.index[:-2]
x_instore = np.arange(len(labels))
x_online = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x_instore - width/2, instore_reg3[1], width, label='In-store West')
rects2 = ax.bar(x_online + width/2, online_reg3[1], width, label='Online West')

ax.set_ylabel('Amount per item per person spend')
ax.set_xlabel('Age')
ax.set_xticks(x_instore)
ax.set_xticklabels(labels.astype(int))
ax.legend()

fig.tight_layout()

plt.show()
```


    
![png](output_55_0.png)
    


### People from West spend in average more amount per item online.

## Comparision of customers age in different regions


```python
age_counts_north = data['age_round'][data['region']==1].value_counts().sort_index()
age_counts_south = data['age_round'][data['region']==2].value_counts().sort_index()
age_counts_east = data['age_round'][data['region']==3].value_counts().sort_index()
age_counts_west = data['age_round'][data['region']==4].value_counts().sort_index()


plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.pie(np.array(age_counts_north.values), labels=age_counts_north.index, autopct='%1.0f%%')
plt.title('North')
plt.subplot(2,2,2)
plt.pie(np.array(age_counts_south.values), labels=age_counts_south.index, autopct='%1.0f%%')
plt.title('South')
plt.subplot(2,2,3)
plt.pie(np.array(age_counts_east.values), labels=age_counts_east.index, autopct='%1.0f%%')
plt.title('East')
plt.subplot(2,2,4)
plt.pie(np.array(age_counts_west.values), labels=age_counts_west.index, autopct='%1.0f%%')
plt.title('West')
plt.show()
```


    
![png](output_58_0.png)
    


### There are no visible outliers. Not all regions have the same age groups.

# Are there differences in the age of customers between regions? <a class="anchor" id="Q3"></a>

### Analyzing figure above we can see that only in South there are people above 80 yo. Apart from that any differences are observed.

# Modeling - Classification <a class="anchor" id="Modeling"></a>

### I will test a lot of different models so I decided to create function to fit, predict and plot results


```python
def fit_predict_plot(model, X_train, X_test, y_train, y_test, xticklabels, yticklabels):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy:', round(accuracy_score(y_test, y_pred)*100), '%')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={"size": 12}, fmt='.1%', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
```


```python
def fit_predict_plot_validation(model, X_train, X_test, y_train, y_test, xticklabels, yticklabels):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = cross_val_score(model, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('Cross Validation accuracy scores: %s' % scores)
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
    print('Accuracy:', round(accuracy_score(y_test, y_pred)*100), '%')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={"size": 12}, fmt='.1%', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
```


```python
def fit_predict_plot_GS(model, X_train, X_test, y_train, y_test, xticklabels, yticklabels, params):
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy:', round(accuracy_score(y_test, y_pred)*100), '%')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={"size": 12}, fmt='.1%', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
```


```python
X = data[['region']]
y = data['age_round']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
age_labels = ['20', '30', '40', '50', '60', '70', '80']
```

## Customer age prediction - 7 bins

### Decision Tree Classifier


```python
model = DecisionTreeClassifier()
fit_predict_plot(model, X_train, X_test, y_train, y_test, age_labels, age_labels)
```

    Accuracy: 25 %
                  precision    recall  f1-score   support
    
              20       0.00      0.00      0.00      1896
              30       0.00      0.00      0.00      3694
              40       0.26      0.81      0.40      4873
              50       0.00      0.00      0.00      3924
              60       0.19      0.32      0.24      3006
              70       0.00      0.00      0.00      1664
              80       0.00      0.00      0.00       938
    
        accuracy                           0.25     19995
       macro avg       0.06      0.16      0.09     19995
    weighted avg       0.09      0.25      0.13     19995
    
    

    C:\Anaconda\envs\ubiqum\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Anaconda\envs\ubiqum\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Anaconda\envs\ubiqum\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


    
![png](output_70_2.png)
    


## Customer age prediction - 3 bins


```python
X = data[['region']]
y = data['age_levels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
age_labels = ['Young','Middle','Old']
```


```python
model = DecisionTreeClassifier()
fit_predict_plot(model, X_train, X_test, y_train, y_test, age_labels, age_labels)
```

    Accuracy: 47 %
                  precision    recall  f1-score   support
    
          Middle       0.46      0.24      0.32      8592
             Old       0.40      0.63      0.49      3208
           Young       0.51      0.65      0.57      8195
    
        accuracy                           0.47     19995
       macro avg       0.46      0.51      0.46     19995
    weighted avg       0.47      0.47      0.45     19995
    
    


    
![png](output_73_1.png)
    


### Models (Decision Tree Classifier and XGBoost Classifier) obtained bigger accuracy using less classes, but it is not sufficient to predict age using only region. An apparent increase in accuracy is expected as the number of classes is reduced. For 7 classes and random classifier, the accuracy is ~14%, while for 3 classes it increases to ~33%.

# If so, can we predict the age of a customer in a region based on other demographic data? <a class="anchor" id="Q4"></a>

### There is no clear correlation between the region and the age of the consumer. Due to this fact I decided to check if adding other data can increase performance of the model.


```python
X = data[['region', 'in-store', 'amount', 'items']]
y = data['age_levels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
age_labels = ['Young','Middle','Old']
```


```python
model = DecisionTreeClassifier()
fit_predict_plot(model, X_train, X_test, y_train, y_test, age_labels, age_labels)
plt.bar(['region', 'in-store', 'amount', 'items'],model.feature_importances_)
plt.show()
```

    Accuracy: 43 %
                  precision    recall  f1-score   support
    
          Middle       0.44      0.44      0.44      8592
             Old       0.33      0.33      0.33      3208
           Young       0.46      0.45      0.46      8195
    
        accuracy                           0.43     19995
       macro avg       0.41      0.41      0.41     19995
    weighted avg       0.43      0.43      0.43     19995
    
    


    
![png](output_78_1.png)
    



    
![png](output_78_2.png)
    



```python
model = xgb.XGBClassifier()
fit_predict_plot(model, X_train, X_test, y_train, y_test, age_labels, age_labels)
xgb.plot_importance(model)
```

    C:\Anaconda\envs\ubiqum\lib\site-packages\xgboost\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [17:54:04] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    Accuracy: 47 %
                  precision    recall  f1-score   support
    
          Middle       0.43      0.46      0.45      8592
             Old       0.39      0.33      0.36      3208
           Young       0.54      0.54      0.54      8195
    
        accuracy                           0.47     19995
       macro avg       0.46      0.44      0.45     19995
    weighted avg       0.47      0.47      0.47     19995
    
    


    
![png](output_79_2.png)
    





    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_79_4.png)
    


### In order to check the validity of features for the models, two charts were prepared. Both models showed that the total amount of purchases is important to predict the age, then to a much lesser extent the number of items purchased, the region and whether the purchase was made in-store or online. Analyzing the presented data, I came to the conclusion that it is not possible to predict the consumer's age on the basis of demographic data.

# Is there any correlation between age of a customer and if the transaction was made online or in the store? Do any other factors predict if a customer will buy online or in our stores? <a class="anchor" id="Q5"></a>

## Trying to predict if the transaction was made online or in the store basic on the age of the customer aggregated to seven levels 


```python
X = data[['age_round']]
y = data['in-store']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
labels = ['in-store', 'online']
```


```python
model = DecisionTreeClassifier(max_depth=4)
fit_predict_plot(model, X_train, X_test, y_train, y_test, labels, labels)
plt.figure(figsize=(16,16))
tree.plot_tree(model)
plt.show()
```

    Accuracy: 59 %
                  precision    recall  f1-score   support
    
               0       0.71      0.28      0.41      9870
               1       0.56      0.89      0.69     10125
    
        accuracy                           0.59     19995
       macro avg       0.63      0.59      0.55     19995
    weighted avg       0.63      0.59      0.55     19995
    
    


    
![png](output_84_1.png)
    



    
![png](output_84_2.png)
    


### The model obtained accuracy 59%. When analyzing the decision tree, it can be noticed that the model correctly recognized that people over 75 years old do not make in-store purchases. This relationship has also been shown previously in the EDA.


```python
model = xgb.XGBClassifier()
fit_predict_plot(model, X_train, X_test, y_train, y_test, age_labels, age_labels)
```

    C:\Anaconda\envs\ubiqum\lib\site-packages\xgboost\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [23:11:32] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    Accuracy: 59 %
                  precision    recall  f1-score   support
    
               0       0.71      0.28      0.41      9870
               1       0.56      0.89      0.69     10125
    
        accuracy                           0.59     19995
       macro avg       0.63      0.59      0.55     19995
    weighted avg       0.63      0.59      0.55     19995
    
    


    
![png](output_86_2.png)
    


### XGBoost also showed similar accuracy as decision trees. Therefore, to obtain an effective model, it is necessary to use more features. It is not possible to predict whether a customer bought online or in a store just based on his age.

## Adding new features for higher accuracy

### At the beginning, I added all the available features to check the importace of the features for the model.


```python
feats = ['age', 'items', 'amount', 'region', 'age_round',
       'amount_per_item', 'age_levels_mean']
X = data[feats]
y = data['in-store']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
labels = ['in-store', 'online']
```


```python
model = DecisionTreeClassifier()
fit_predict_plot(model, X_train, X_test, y_train, y_test, labels, labels)
plt.figure(figsize=(12,6))
plt.bar(feats,model.feature_importances_)
plt.show()
```

    Accuracy: 84 %
                  precision    recall  f1-score   support
    
               0       0.84      0.85      0.84      9870
               1       0.85      0.84      0.84     10125
    
        accuracy                           0.84     19995
       macro avg       0.84      0.84      0.84     19995
    weighted avg       0.84      0.84      0.84     19995
    
    


    
![png](output_91_1.png)
    



    
![png](output_91_2.png)
    


## What about items? Are these even in the tree? Why or why not? <a class="anchor" id="Q4"></a>

### Items are not in the tree and has very low importance for models due to its low correlation with target variable


```python
model = xgb.XGBClassifier()
fit_predict_plot(model, X_train, X_test, y_train, y_test, labels, labels)
xgb.plot_importance(model)
```

    C:\Anaconda\envs\ubiqum\lib\site-packages\xgboost\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [18:09:48] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    Accuracy: 88 %
                  precision    recall  f1-score   support
    
               0       0.98      0.79      0.87      9870
               1       0.82      0.98      0.90     10125
    
        accuracy                           0.88     19995
       macro avg       0.90      0.88      0.88     19995
    weighted avg       0.90      0.88      0.88     19995
    
    


    
![png](output_94_2.png)
    





    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_94_4.png)
    


### Both models showed a high efficiency of over 80%. In the case of decision trees, the highest importance was given to features such as region, amount, age and amount_per_item. The Items feature is not very important for the model. For further modeling, I used features such as region, amount, age.


```python
feats = ['age', 'amount', 'region']
X = data[feats]
y = data['in-store']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
age_labels = ['in-store', 'online']
```


```python
model = DecisionTreeClassifier()
fit_predict_plot_validation(model, X_train, X_test, y_train, y_test, labels, labels)
plt.figure(figsize=(12,6))
plt.bar(feats,model.feature_importances_)
plt.show()
```

    Cross Validation accuracy scores: [0.85397566 0.84547425 0.84280713 0.84914152 0.84761587 0.84978326
     0.84544848 0.84344782 0.84544848 0.84694898]
    Cross Validation accuracy: 0.847 +/- 0.003
    Accuracy: 85 %
                  precision    recall  f1-score   support
    
               0       0.84      0.85      0.84      9870
               1       0.85      0.84      0.85     10125
    
        accuracy                           0.85     19995
       macro avg       0.85      0.85      0.85     19995
    weighted avg       0.85      0.85      0.85     19995
    
    


    
![png](output_97_1.png)
    



    
![png](output_97_2.png)
    



```python
model = xgb.XGBClassifier(use_label_encoder=False)
fit_predict_plot_validation(model, X_train, X_test, y_train, y_test, labels, labels)
xgb.plot_importance(model)
```

    [18:09:54] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:09:57] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:00] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:03] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:05] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:08] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:11] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:14] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:17] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:20] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [18:10:23] WARNING: ..\src\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    Cross Validation accuracy scores: [0.88681447 0.885981   0.89098183 0.88348058 0.8882961  0.88546182
     0.88746249 0.88446149 0.88629543 0.88179393]
    Cross Validation accuracy: 0.886 +/- 0.002
    Accuracy: 89 %
                  precision    recall  f1-score   support
    
               0       0.98      0.79      0.87      9870
               1       0.83      0.98      0.90     10125
    
        accuracy                           0.89     19995
       macro avg       0.90      0.88      0.88     19995
    weighted avg       0.90      0.89      0.88     19995
    
    


    
![png](output_98_1.png)
    





    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_98_3.png)
    



```python
model = RandomForestClassifier()
fit_predict_plot_validation(model, X_train, X_test, y_train, y_test, labels, labels)
plt.figure(figsize=(12,6))
plt.bar(feats,model.feature_importances_)
plt.show()
```

    Cross Validation accuracy scores: [0.85414236 0.84730788 0.84647441 0.85364227 0.85328443 0.85128376
     0.85645215 0.84911637 0.84794932 0.84994998]
    Cross Validation accuracy: 0.851 +/- 0.003
    Accuracy: 85 %
                  precision    recall  f1-score   support
    
               0       0.85      0.84      0.85      9870
               1       0.85      0.85      0.85     10125
    
        accuracy                           0.85     19995
       macro avg       0.85      0.85      0.85     19995
    weighted avg       0.85      0.85      0.85     19995
    
    


    
![png](output_99_1.png)
    



    
![png](output_99_2.png)
    



```python
tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8]}
fit_predict_plot_GS(DecisionTreeClassifier(), X_train, X_test, y_train, y_test, labels, labels, tree_para)
```

    Accuracy: 89 %
                  precision    recall  f1-score   support
    
               0       1.00      0.77      0.87      9870
               1       0.82      1.00      0.90     10125
    
        accuracy                           0.89     19995
       macro avg       0.91      0.89      0.89     19995
    weighted avg       0.91      0.89      0.89     19995
    
    


    
![png](output_100_1.png)
    



```python
def XGBoost_fit_predict_CV(X_train, X_test, y_train, y_test):
    param_grid_cxgb = params = {
        'min_child_weight': [1, 5, 10, 20, 30],
        'gamma': [0.3, 0.5, 1, 1.5, 2, 5],
        'colsample_bytree': [0.6, 0.8, 1.0, 1.5, 2.0],
        'max_depth': [3, 4, 5, 6, 8, 10]
        }
    eval_s = [(X_train_val, y_train_val),(X_val,y_val)]
    cxgb_reg = xgb.XGBClassifier(learning_rate=0.08, n_estimators=300, objective='binary:logistic',
                        silent=True, nthread=1, use_label_encoder=False);
    cxgb = RandomizedSearchCV(cxgb_reg, param_grid_cxgb, cv=5, scoring='recall',verbose=3);
    cxgb.fit(X_train_val, y_train_val,eval_set=eval_s, early_stopping_rounds=20)
    print(cxgb.best_params_)
    print(cxgb.best_estimator_)
    xgb_res = cxgb.cv_results_
    xgb_best = cxgb.best_estimator_
    results = xgb_best.evals_result()
    plt.figure()
    plt.plot(results['validation_0']['logloss'], label='train')
    plt.plot(results['validation_1']['logloss'], label='test')
    plt.legend(['Train set', 'Validation set'])
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.show()
    y_pred = xgb_best.predict(X_test)
    print('Accuracy:', round(accuracy_score(y_test, y_pred)*100), '%')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={"size": 12}, fmt='.1%', cbar=False, xticklabels=['In-store','Online'], yticklabels=['In-store','Online'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    plt.figure()
    xgb.plot_importance(xgb_best)
    plt.show()
    return round(accuracy_score(y_test, y_pred)*100), cxgb.best_estimator_
```


```python
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=123)
```


```python
acc, best_est = XGBoost_fit_predict_CV(X_train, X_test, y_train, y_test)
```


```python
feats_sets = [['age', 'amount_normalized', 'region'],
              ['age_levels_5_mean', 'amount', 'region'], 
              ['age_levels_5_mean', 'amount_levels_5_mean', 'region'], 
              ['age', 'amount_levels_5_mean', 'region'], 
              ['age', 'Region_2', 'Region_3', 'Region_4', 'amount_normalized'], 
              ['age_levels_5_mean', 'Region_2', 'Region_3', 'Region_4', 'amount'], 
              ['age_levels_5_mean', 'Region_2', 'Region_3', 'Region_4', 'amount_levels_5_mean']]
results_feats = {}

```


```python
for i, feats in enumerate(feats_sets):
    X = data[feats]
    y = data['in-store']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=123)
    accuracy, best_model = XGBoost_fit_predict_CV(X_train, X_test, y_train, y_test)
    results_feats[i] = {'feats':feats, 'accuracy':accuracy, 'best_model':best_model}
    
```


```python
def fit_predict_plot_GS_acc(model, X_train, X_test, y_train, y_test, xticklabels, yticklabels, params):
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy:', round(accuracy_score(y_test, y_pred)*100), '%')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={"size": 12}, fmt='.1%', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    return round(accuracy_score(y_test, y_pred)*100), clf.best_params_
```


```python
results_feats = {}
tree_para = {'criterion':['gini','entropy'],'max_depth':[2,3,4,5,6,7,8, 10,20]}

for i, feats in enumerate(feats_sets):
    X = data[feats]
    y = data['in-store']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
    model = DecisionTreeClassifier()
    accuracy, best_params = fit_predict_plot_GS_acc(model, X_train, X_test, y_train, y_test, labels, labels, tree_para)
    results_feats[i] = {'feats':feats, 'accuracy':accuracy, 'best_model': best_params}
```

    Accuracy: 89 %
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.87      9870
               1       0.82      1.00      0.90     10125
    
        accuracy                           0.89     19995
       macro avg       0.91      0.89      0.89     19995
    weighted avg       0.91      0.89      0.89     19995
    
    


    
![png](output_107_1.png)
    


    Accuracy: 88 %
                  precision    recall  f1-score   support
    
               0       1.00      0.75      0.86      9870
               1       0.81      1.00      0.89     10125
    
        accuracy                           0.88     19995
       macro avg       0.90      0.88      0.88     19995
    weighted avg       0.90      0.88      0.88     19995
    
    


    
![png](output_107_3.png)
    


    Accuracy: 87 %
                  precision    recall  f1-score   support
    
               0       0.96      0.76      0.85      9870
               1       0.81      0.97      0.88     10125
    
        accuracy                           0.87     19995
       macro avg       0.88      0.87      0.87     19995
    weighted avg       0.88      0.87      0.87     19995
    
    


    
![png](output_107_5.png)
    


    Accuracy: 88 %
                  precision    recall  f1-score   support
    
               0       0.96      0.79      0.86      9870
               1       0.82      0.96      0.89     10125
    
        accuracy                           0.88     19995
       macro avg       0.89      0.88      0.88     19995
    weighted avg       0.89      0.88      0.88     19995
    
    


    
![png](output_107_7.png)
    


    Accuracy: 89 %
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.87      9870
               1       0.82      0.99      0.90     10125
    
        accuracy                           0.89     19995
       macro avg       0.91      0.89      0.89     19995
    weighted avg       0.91      0.89      0.89     19995
    
    


    
![png](output_107_9.png)
    


    Accuracy: 88 %
                  precision    recall  f1-score   support
    
               0       1.00      0.75      0.86      9870
               1       0.81      1.00      0.89     10125
    
        accuracy                           0.88     19995
       macro avg       0.90      0.88      0.88     19995
    weighted avg       0.90      0.88      0.88     19995
    
    


    
![png](output_107_11.png)
    


    Accuracy: 87 %
                  precision    recall  f1-score   support
    
               0       0.96      0.76      0.85      9870
               1       0.81      0.97      0.88     10125
    
        accuracy                           0.87     19995
       macro avg       0.88      0.87      0.87     19995
    weighted avg       0.88      0.87      0.87     19995
    
    


    
![png](output_107_13.png)
    



```python
results_feats = {}
tree_para = {
 'max_depth': [3, 4, 5, 6, 9, None],
 'n_estimators': [100, 200, 400]}

for i, feats in enumerate(feats_sets):
    X = data[feats]
    y = data['in-store']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
    model = RandomForestClassifier()
    accuracy, best_params = fit_predict_plot_GS_acc(model, X_train, X_test, y_train, y_test, labels, labels, tree_para)
    print(feats)
    results_feats[i] = {'feats':feats, 'accuracy':accuracy, 'best_model': best_params}
```

    Accuracy: 89 %
                  precision    recall  f1-score   support
    
               0       1.00      0.78      0.87      9870
               1       0.82      1.00      0.90     10125
    
        accuracy                           0.89     19995
       macro avg       0.91      0.89      0.89     19995
    weighted avg       0.91      0.89      0.89     19995
    
    


    
![png](output_108_1.png)
    


    ['age', 'amount_normalized', 'region']
    Accuracy: 88 %
                  precision    recall  f1-score   support
    
               0       1.00      0.75      0.86      9870
               1       0.81      1.00      0.89     10125
    
        accuracy                           0.88     19995
       macro avg       0.90      0.88      0.88     19995
    weighted avg       0.90      0.88      0.88     19995
    
    


    
![png](output_108_3.png)
    


    ['age_levels_5_mean', 'amount', 'region']
    Accuracy: 87 %
                  precision    recall  f1-score   support
    
               0       0.96      0.76      0.85      9870
               1       0.81      0.97      0.88     10125
    
        accuracy                           0.87     19995
       macro avg       0.88      0.87      0.87     19995
    weighted avg       0.88      0.87      0.87     19995
    
    


    
![png](output_108_5.png)
    


    ['age_levels_5_mean', 'amount_levels_5_mean', 'region']
    Accuracy: 88 %
                  precision    recall  f1-score   support
    
               0       0.95      0.79      0.86      9870
               1       0.82      0.96      0.89     10125
    
        accuracy                           0.88     19995
       macro avg       0.89      0.88      0.87     19995
    weighted avg       0.89      0.88      0.88     19995
    
    


    
![png](output_108_7.png)
    


    ['age', 'amount_levels_5_mean', 'region']
    Accuracy: 89 %
                  precision    recall  f1-score   support
    
               0       1.00      0.77      0.87      9870
               1       0.82      1.00      0.90     10125
    
        accuracy                           0.89     19995
       macro avg       0.91      0.89      0.89     19995
    weighted avg       0.91      0.89      0.89     19995
    
    


    
![png](output_108_9.png)
    


    ['age', 'Region_2', 'Region_3', 'Region_4', 'amount_normalized']
    Accuracy: 88 %
                  precision    recall  f1-score   support
    
               0       1.00      0.75      0.86      9870
               1       0.81      1.00      0.89     10125
    
        accuracy                           0.88     19995
       macro avg       0.90      0.88      0.88     19995
    weighted avg       0.90      0.88      0.88     19995
    
    


    
![png](output_108_11.png)
    


    ['age_levels_5_mean', 'Region_2', 'Region_3', 'Region_4', 'amount']
    Accuracy: 87 %
                  precision    recall  f1-score   support
    
               0       0.96      0.76      0.85      9870
               1       0.81      0.97      0.88     10125
    
        accuracy                           0.87     19995
       macro avg       0.88      0.87      0.87     19995
    weighted avg       0.88      0.87      0.87     19995
    
    


    
![png](output_108_13.png)
    


    ['age_levels_5_mean', 'Region_2', 'Region_3', 'Region_4', 'amount_levels_5_mean']
    

# Conclusions <a class="anchor" id="conclusions"></a>

### The dataset contained 21 duplicates which were removed but do not contain missing data. During the analysis of the collection, it was noticed that customers from the north only buy in-store, while customers from the south choose online shopping. Customers from other regions buy both online and in-store. There was no correlation between items and amount, but there were some relationships between other variables such as: age - in-store, region - items, region - age. Total number of items purchased online is same as total number of items purchased online, but customers spend more money during online shoping, because they choose more expensive products than in in-store shopping or it is effect of added shipping costs to amout of items. It may also be due to the higher price of products in the online store. In each age group, customers buy on average the same number of products per person, but younger peoples spend more money than older. Additional analyzes show that people older than 80 years old do not buy in-store, only people in their seventies bought more expensive products in-store than online, people from 30-60 years old spend more less the same in-store and online per one product and youngest people spend more per item online than in-store. Analyzes have shown that customers spend more online. Overall, there were large differences between regions. Both taking into account the type of purchases and the age groups of customers. Due to these differences, it would be necessary to carry out detailed analyzes for each of the regions separately. One example would be to find out why in the North and South regions customers only shop in-store or only online. This may be due to the lack of availability of another purchasing method. Another issue to be clarified could be the shipping cost included in the amount of online purchases (the question is whether it is added, how much is it, etc.), and then preparing the data without shipping costs. 

### Many experiments have been carried out with the model's hyperparameters and with the use of various features, but it did not bring any significant improvement. Also, applying the encoder to the region feature did not increase the efficiency of the model. However, this allows the variable to be treated as categorical but nominal. The conducted tests show that we are not able to correctly predict the age of the client based on demographic data. However, it is possible to predict whether the customer will buy online or in-store with an efficiency of 89%. The models show the same effectiveness on training and test data, which means that there was no overfitting phenomenon and the models can be used to predict new data. Region and amount spent are of the greatest importance in predicting this. The number of items purchased is of low importance. Variable bining has no influence on the achieved efficacy value.

 |Features | Target | Classifier | Hyperparams. | Accuracy |
 |:------|------|------|------|------|
 |Region | Age (7 bins) | DecisionTree| Default | 25% |
 |Region | Age (7 bins) | XBG | Default | 25% |
 |Region | Age (3 bins) | DecisionTree | Default | 47% |
 |Region | Age (3 bins) | XBG| Default | 47% |
 |Region,In-store, Amount, Items | Age (3 bins) | DecisionTree | Default | 43% |
 |Region,In-store, Amount, Items | Age (3 bins) | XBG | Default | 47% |
 |Age (7 bins) | In-store | DecisionTree | max_depth=4 | 59% |
 |Age (7 bins)  | In-store | XBG | Default | 59% |
 |Age, Items, Amount, Region, Age (7 bins), Amount per item, Age (3 bins) | In-store | DecisionTree | default | 84% |
 |Age, Items, Amount, Region, Age (7 bins), Amount per item, Age (3 bins)  | In-store | XBG | Default | 88% |
 |Age, Amount, Region | In-store | DecisionTree | Default | 84% |
 |Age, Amount, Region | In-store | DecisionTree | criterion=gini, max_depth=8| 84% |
 |Age, Amount, Region | In-store | XBG | Default | 89% |
 |Age, Amount, Region | In-store | RandomForest | Default | 85% |
 |Age, Amount normalized, Region | In-store | DecisionTree | criterion=entropy, max_depth=10| 89% |
 |Age (3 bins), Amount (5 bins), Region (One hot encoded) | In-store | DecisionTree | criterion=gini, max_depth=8| 89% |
 |Age, Amount normalized, Region | In-store | XGB | n_estimators=300, booster='gbtree', max_depth=5| 89% |
 |Age (3 bins), Amount (5 bins), Region (One hot encoded) | In-store | XGB | n_estimators=300, booster='gbtree', max_depth=3 | 89% |
 |Age (5 bins), Amount, Region | In-store | RandomForest |n_estimators=200, max_depth=5  | 87% |
 |Age (5 bins), Amount, Region (One hot encoded)| In-store | RandomForest |n_estimators=400, max_depth=3 | 88% |


```python

```
