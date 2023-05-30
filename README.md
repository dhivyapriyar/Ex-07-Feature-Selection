# Ex-07-Feature-Selection
## AIM

To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation

Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM

### STEP 1
Read the given Data

### STEP 2
Clean the Data Set using Data Cleaning Process

### STEP 3
Apply Feature selection techniques to all the features of the data set

### STEP 4
Save the data to the file


# CODE

NAME: DHIVYAPRIYA.R

REGISTER NO,; 212222230032

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1)

y = df1["Survived"]

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

p= []

X_1 = X[cols]

X_1 = sm.add_constant(X_1)

model = sm.OLS(y,X_1).fit()

p = pd.Series(model.pvalues.values[1:],index = cols)  

pmax = max(p)

feature_with_p_max = p.idxmax()

if(pmax>0.05):

    cols.remove(feature_with_p_max)
    
else:

    break
    selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)

high_score=0

nof=0

score_list =[]

for n in range(len(nof_list)):

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

model = LinearRegression()

rfe = RFE(model,step=nof_list[n])

X_train_rfe = rfe.fit_transform(X_train,y_train)

X_test_rfe = rfe.transform(X_test)

model.fit(X_train_rfe,y_train)

score = model.score(X_test_rfe,y_test)

score_list.append(score)

if(score>high_score):

    high_score = score
    
    nof = nof_list[n]
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()

# OUPUT
![Screenshot 2023-05-23 112556](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/96adaf5b-f980-475a-adf7-dc398c43ca0f)
![Screenshot 2023-05-23 112627](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/6996c5d6-f9c3-4afe-a8c0-ce7a84625a7c)
![Screenshot 2023-05-23 112635](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/23c467a7-3ce0-4651-a7e5-c99c057cd7cc)
![Screenshot 2023-05-23 112645](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/d6c1dab5-b53e-4c9a-bbbc-7058c10f1232)
![Screenshot 2023-05-23 112702](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/757bd557-54c1-4fef-9494-e583f8e3f01a)
![Screenshot 2023-05-23 112725](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/1e63eecc-d317-4307-b86e-ae45812d9779)
![Screenshot 2023-05-23 112749](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/f14241ef-361b-4322-b289-a09a74316190)
![Screenshot 2023-05-23 112801](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/cca33855-7024-49b9-8021-378af0d74e4c)
![Screenshot 2023-05-23 112816](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/54d29003-a61c-4a63-8da8-47318461035b)
![Screenshot 2023-05-23 112827](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/26120292-25a6-4350-ba92-03d76fe39353)
![Screenshot 2023-05-23 112845](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/80e6d621-22e5-4dcf-bcdc-c4f3880663e8)
![Screenshot 2023-05-23 112942](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/6eadcf2e-b6df-41bb-91ac-b363366b7d6d)
![Screenshot 2023-05-23 112954](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/fe139a3f-32e1-484b-9906-67937f92ca79)
![Screenshot 2023-05-23 113023](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/b2665986-e633-4c19-a051-1d7e273ceda0)
![Screenshot 2023-05-23 113035](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/f8992836-f71d-4499-ad73-9adfc44e1d08)
![Screenshot 2023-05-23 113043](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/5415aa20-5a12-44a9-ba64-3ac44dd91e3b)
![Screenshot 2023-05-23 113055](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/be8a8bd4-5e1b-43b7-a894-42e8c5d5b2d5)
![Screenshot 2023-05-23 113113](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/6bae7859-ac3c-4799-aa84-0434a44e622a)
![Screenshot 2023-05-23 113204](https://github.com/dhivyapriyar/Ex-07-Feature-Selection/assets/119477552/240f3023-174e-453c-9141-09848f91a16f)


