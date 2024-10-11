# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/7ec14dd8-7a19-46ed-8ef5-111d32cd0ea8)
```python
df.dropna()
```
![image](https://github.com/user-attachments/assets/7a643efd-e783-4e6f-8bcd-6500ae9f631d)
```python
# TYPE CODE TO FIND MAXIMUM VALUE FROM HEIGHT AND WEIGHT FEATURE

max_height = df['Height'].max()
max_weight = df['Weight'].max()

print("Maximum Height:",max_height)
print("Maximum Weight:",max_weight)
```
![image](https://github.com/user-attachments/assets/19caaaae-c6f2-4912-866e-3832afdd3b7b)
```python
#Perform minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/ea1da995-4900-42d9-a9d4-dc1ac731b922)
```python
#Perform standard scaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/062c2843-f0ca-406a-8f96-5f2a5dbf0e2e)
```python
#Perform Normalizer

from sklearn.preprocessing import Normalizer

scaler = Normalizer()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/7dad27ec-7682-4eb8-966b-b46a9d302d67)
```python
#Perform MaxAbsScaler

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/49642099-8b87-4153-8ea5-8130258343d6)
```python
#Perform RobustScaler

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/1aadcf31-8a86-4caf-94a1-4002ff867058)
```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```
```python
data=pd.read_csv('income.csv')
data.head()
```
![image](https://github.com/user-attachments/assets/82210c97-1533-4dca-b813-b282dc8111c9)
```python
data.columns
```
![image](https://github.com/user-attachments/assets/ce387064-16f0-4b8a-bed2-9070b15beb1b)
```python
data.shape
```
![image](https://github.com/user-attachments/assets/787b73a1-8a27-42f8-b939-46daea7fec80)
```python
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a7da2fe4-f229-40ab-9049-aa747e6ef741)
```python
# feature matrix

X = data.drop('SalStat',axis=1)
y = data['SalStat']
print(y)
```
![image](https://github.com/user-attachments/assets/66a89271-2546-4289-bd1d-9f38cf61da98)
```python
sal=data["SalStat"]

data["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data['SalStat'])
```
![image](https://github.com/user-attachments/assets/54443c52-514a-445e-8a77-2f2ed8f27bcd)
```python
sal2=data['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/952ca00d-3876-45e5-ac2b-ad61a25ff4b2)
```python
new_data=pd.get_dummies(data, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/0920a44a-8ff1-425e-aea3-121976addb22)
```python
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/7cbaeb03-37b0-429a-a4fa-281326632532)
```python
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/0184b61e-48bc-43aa-a698-d30cd80c4430)
```python
a=new_data['SalStat'].values
print(a)
```
![image](https://github.com/user-attachments/assets/0c2123db-a9c9-454b-86e1-2454a5842213)
```python
b=new_data[features].values
print(b)
```
![image](https://github.com/user-attachments/assets/44b633db-9515-42a8-a70b-b4edf696b8bd)
```python
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/7772df84-e59b-4760-a519-0180df299c66)
```python
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/0e1c372a-70e6-4839-a13f-4197054c31c9)
```python
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/ff63393e-d65f-4354-a2ff-344d380f1bd4)
```python
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/487ecf72-9cba-47df-a457-c1e2d31b1d90)
```python
data.shape
```
![image](https://github.com/user-attachments/assets/fd6af5f0-7ed1-44b1-a249-ce0924170432)

# RESULT:
 The Feature scaling and feature selection executed successfully for the given data.
