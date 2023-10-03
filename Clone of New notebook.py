# -*- coding: utf-8 -*-

# -- Sheet --

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('IRIS.csv')
print(df.head())

print(df.shape)
print(df.info())

print(df.isnull().sum())
print(df.describe())

plt.figure(figsize=(8, 6))
sns.countplot(x='species', data=df)
plt.xlabel('Species', fontsize=14)
plt.show()

df['species'].value_counts()
sns.set(rc={'figure.figsize': (10, 8)})
sns.distplot(df["sepal_length"], kde=True, color="red", bins=10)

plt.figure(figsize=(8, 6))
sns.scatterplot(x="sepal_length", y=df.index, data=df)
plt.show()

sns.set(rc={"figure.figsize": (8, 6)})
sns.distplot(df["sepal_width"], kde=True, color="navy", bins=10)

plt.figure(figsize=(10, 8))
sns.scatterplot(x="sepal_width", y=df.index, data=df)
plt.show()

sns.set(rc={'figure.figsize': (8, 6)})
sns.distplot(df['petal_length'], kde=True, color="blue", bins=10)

plt.figure(figsize=(8, 8))
sns.scatterplot(x="petal_length", y=df.index, data=df)
plt.show()

sns.set(rc={"figure.figsize": (10, 8)})
sns.distplot(df["petal_width"], kde=True, color="orange", bins=10)

plt.figure(figsize=(10, 8))
sns.scatterplot(x="petal_width", y=df.index, data=df)
plt.show()

duplicate = df.duplicated()
print(duplicate.sum())

df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

print(df.isnull().sum())

num_cols = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(16, 14))
sns.boxplot(num_cols)
plt.show()

def remove_outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

for i in num_cols.columns:
    lower_range, upper_range = remove_outlier(df[i])
    df[i] = np.where(df[i] > upper_range, upper_range, df[i])
    df[i] = np.where(df[i] < lower_range, lower_range, df[i])

num_cols = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(14, 12))
sns.boxplot(num_cols)
plt.show()

print(df.info())

plt.figure(figsize=(18, 16))
sns.histplot(data=df, x="sepal_length", hue='species', kde=True, bins=20)
plt.title('species By Sepal_Length')
plt.xlabel('Sepal_length')
plt.ylabel('Count')
plt.legend(title='species', loc='upper right', labels=['No Species', 'species'])
plt.show()

plt.figure(figsize=(18, 16))
sns.histplot(data=df, x="sepal_width", hue='species', kde=True, bins=20)
plt.title('species By Sepal_width')
plt.xlabel('Sepal_width')
plt.ylabel('Count')
plt.legend(title='species', loc='upper right', labels=['No Species', 'species'])
plt.show()

plt.figure(figsize=(18, 16))
sns.histplot(data=df, x="petal_length", hue='species', kde=True, bins=20)
plt.title('species By petal_Length')
plt.xlabel('petal_length')
plt.ylabel('Count')
plt.legend(title='species', loc='upper right', labels=['No Species', 'species'])
plt.show()

plt.figure(figsize=(18, 16))
sns.histplot(data=df, x="petal_width", hue='species', kde=True, bins=20)
plt.title('species By petal_width')
plt.xlabel('petal_width')
plt.ylabel('Count')
plt.legend(title='species', loc='upper right', labels=['No Species', 'species'])
plt.show()

sns.scatterplot(x="sepal_length", y="petal_length", data=df, hue="species")
plt.show()

sns.scatterplot(x="sepal_width", y="petal_width", data=df, hue="species")
plt.show()

sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)

sns.pairplot(df, hue="species")
plt.show()

def plots(num_cols, variable):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)  # num_cols[variable].hist()
    sns.distplot(num_cols[variable], kde=True, bins=10)
    plt.title(variable)
    plt.subplot(1, 2, 2)
    stats.probplot(num_cols[variable], dist="norm", plot=pylab)
    plt.title(variable)
    plt.show()

for i in num_cols.columns:
    plots(num_cols, i)

X = df.iloc[:,:4]
Y = df['species']
X.head()

mi_score = mutual_info_classif(X,Y)
mi_score = pd.Series(mi_score)
mi_score.index = X.columns
mi_score.sort_values(ascending=True)

mi_score.sort_values(ascending=True).plot.bar(figsize=(12,10))

train_data,test_data,train_label,test_label = train_test_split(X,Y,test_size=0.2,random_state=0)
print("train_data :",train_data.shape)
print("train_label :",train_label.shape)
print("test_data :",test_data.shape)
print("test_label :",test_label.shape)

sc = StandardScaler()
train_data_sc = sc.fit_transform(train_data)
test_data_sc = sc.fit_transform(test_data)

print(train_data_sc)

model_lr = LogisticRegression().fit(train_data_sc, train_label)

y_pred_1 = model_lr.predict(test_data_sc)
print(y_pred_1)

accuracy_score(y_pred_1,test_label)

confusion_matrix(y_pred_1,test_label)

confusion_matrix(y_pred_1,test_label)

confusion_matrix(y_pred_1,test_label)

print(classification_report(y_pred_1,test_label))

model_rf = RandomForestClassifier().fit(train_data_sc,train_label)

y_pred_2 = model_rf.predict(test_data_sc)

print("Train Data Accuracy :",(model_rf.score(train_data_sc,train_label)))
print("Test Data Accuracy :",(accuracy_score(y_pred_2,test_label)))

confusion_matrix(y_pred_2,test_label)

print(classification_report(y_pred_2,test_label))

model_knn = KNeighborsClassifier(n_neighbors=3).fit(train_data_sc,train_label)

y_pred_3 = model_knn.predict(test_data_sc)

print("Test Data Accuracy :",(accuracy_score(y_pred_3,test_label)))

confusion_matrix(y_pred_3,test_label)

print(classification_report(y_pred_3,test_label))

# -- Sheet 2 --

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('StudentsPerformance.csv')
df.head()

df.tail()

df.shape

df.columns

df.dtypes

df['lunch'].unique()

df.nunique()

df.describe()

df['gender'].value_counts()

df.isnull()

df.isnull().sum()

sns.heatmap(df.isnull())
plt.show()

preparation = (len(df[df['test preparation course'] == 'completed']) / len(df)) * 100
preparation

average_score = df['math score'].mean()
average_score

df['test preparation course'].value_counts().plot(kind = 'bar')

plt.figure(figsize=(10,8))
sns.countplot(data=df, x='gender', hue ='test preparation course')
plt.xlabel('gender')
plt.ylabel('test preparation course')
plt.title('Course Completion By Gender')
plt.show()

df['lunch'].value_counts().plot(kind = 'bar')

df['lunch'].value_counts()

lunch_avg_gender = df.groupby('lunch')['gender'].value_counts().plot(kind = 'bar')
lunch_avg_gender

gender_avg_score = df.groupby('gender')['reading score'].mean()
gender_avg_score

df.gender.value_counts().plot(kind = 'bar')

level_edu = df['parental level of education'].value_counts()
level_edu

# -- Sheet 3 --

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif,SelectKBest,f_classif
from sklearn.model_selection import  train_test_split,cross_val_score

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve,auc

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()

df.shape

df.info()

df.isnull().sum()

df.describe()

df.info()

plt.figure(figsize=(10,8))
sns.countplot(x = 'SeniorCitizen', data = df , palette = 'mako')
plt.xlabel('SeniorCitizen', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'gender', data = df , palette = 'mako')
plt.xlabel('gender', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'Partner', data = df , palette = 'mako')
plt.xlabel('Partner', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'Dependents', data = df , palette = 'mako')
plt.xlabel('Dependents', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'PhoneService', data = df , palette = 'mako')
plt.xlabel('PhoneService', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'MultipleLines', data = df , palette = 'mako')
plt.xlabel('MultipleLines', fontsize= 14)
plt.show()

df["MultipleLines"].value_counts()

df['MultipleLines'].replace("No phonw service", "No", inplace = True)

plt.figure(figsize=(10,8))
sns.countplot(x = 'MultipleLines', data = df , palette = 'mako')
plt.xlabel('MultipleLines', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'InternetService', data = df , palette = 'mako')
plt.xlabel('InternetService', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'OnlineSecurity', data = df , palette = 'mako')
plt.xlabel('OnlineSecurity', fontsize= 14)
plt.show()

df["OnlineSecurity"].replace("No Internet Service ","No", inplace = True)

plt.figure(figsize=(10,8))
sns.countplot(x = 'OnlineSecurity', data = df , palette = 'mako')
plt.xlabel('OnlineSecurity', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'OnlineBackup', data = df , palette = 'mako')
plt.xlabel('OnlinBackup', fontsize= 14)
plt.show()

df["OnlineBackup"].replace("No Internet Service ","No", inplace = True)

plt.figure(figsize=(10,8))
sns.countplot(x = 'OnlineBackup', data = df , palette = 'mako')
plt.xlabel('OnlinBackup', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'DeviceProtection', data = df , palette = 'mako')
plt.xlabel('DeviceProtection', fontsize= 14)
plt.show()

df["DeviceProtection"].replace("No Internet Service ","No", inplace = True)

plt.figure(figsize=(10,8))
sns.countplot(x = 'DeviceProtection', data = df , palette = 'mako')
plt.xlabel('DeviceProtection', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'TechSupport', data = df , palette = 'mako')
plt.xlabel('Techsupport', fontsize= 14)
plt.show()

df["TechSupport"].replace("No Internet Service ","No", inplace = True)

plt.figure(figsize=(10,8))
sns.countplot(x = 'TechSupport', data = df , palette = 'mako')
plt.xlabel('Techsupport', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'StreamingTV', data = df , palette = 'mako')
plt.xlabel('StreamingTV', fontsize= 14)
plt.show()

df["StreamingTV"].replace("No Internet Service ","No", inplace = True)

plt.figure(figsize=(10,8))
sns.countplot(x = 'StreamingTV', data = df , palette = 'mako')
plt.xlabel('StreamingTV', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'StreamingMovies', data = df , palette = 'mako')
plt.xlabel('StreamingMovies', fontsize= 14)
plt.show()

df["StreamingMovies"].replace("No Internet Service ","No", inplace = True)

plt.figure(figsize=(10,8))
sns.countplot(x = 'StreamingMovies', data = df , palette = 'mako')
plt.xlabel('StreamingMovies', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'Contract', data = df , palette = 'mako')
plt.xlabel('Contract', fontsize= 14)
plt.show()

df["Contract"].replace("No Internet Service ","No", inplace = True)

plt.figure(figsize=(10,8))
sns.countplot(x = 'Contract', data = df , palette = 'mako')
plt.xlabel('Contract', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'PaperlessBilling', data = df , palette = 'mako')
plt.xlabel('PaperlessBilling', fontsize= 14)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = 'PaymentMethod', data = df , palette = 'mako')
plt.xlabel('PaymentMethod', fontsize= 14)
plt.show()

df["TotalCharges"].value_counts()

df["TotalCharges"].replace(" ",np.nan, inplace = True)

df["TotalCharges"].value_counts()

df["TotalCharges"] = df["TotalCharges"].astype("float64")

plt.figure(figsize=(10,8))
sns.countplot(x = 'Churn', data = df , palette = 'mako')
plt.xlabel('Churn', fontsize= 14)
plt.show()

sns.set(rc={"figure.figsize": (8,6)})
sns.displot(df["tenure"], kde=True, color="orange", bins=10)

sns.set(rc={"figure.figsize": (8,6)})
sns.displot(df["MonthlyCharges"], kde=True, color="red", bins=10)

sns.set(rc={"figure.figsize": (8,6)})
sns.displot(df["TotalCharges"], kde=True, color="blue", bins=10)

sns.set(rc={"figure.figsize": (8,6)})
sns.displot(df["tenure"], kde=True, color="navy", bins=10)

df.isnull().sum()

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

df.isnull().sum()

duplicate = df.duplicated()
print(duplicate.sum())

num_cols = df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(15,10))
sns.boxplot(data=num_cols)
plt.show()

plt.boxplot(df['SeniorCitizen'])
plt.show()

plt.boxplot(df['tenure'])
plt.show()

df["SeniorCitizen"].value_counts()

df.info

plt.figure(figsize=(8,6))
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Churn byGender")
plt.xlabel("Gender")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='Churn', hue='SeniorCitizen', data=df)
plt.title("Churn by SeniorCitizen")
plt.xlabel("Churn")
plt.ylabel("COunt")
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='PhoneService', hue='Churn', data=df)
plt.title("Churn By PhoneService")
plt.xlabel("Gender")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='Dependents', hue='Churn', data=df)
plt.title("Churn by Dependents")
plt.xlabel("Dependents")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='MultipleLines', hue='Churn', data=df)
plt.title("Churn By MultipleLines")
plt.xlabel("MultipleLines")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title("Churn by InternetService")
plt.xlabel("InternetService")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='OnlineSecurity', hue='Churn', data=df)
plt.title("Churn by OnlineSecurity")
plt.xlabel("OnlineSecurity")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='OnlineBackup', hue='Churn', data=df)
plt.title("Churn by OnlineBackup")
plt.xlabel("OnlineBackup")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='DeviceProtection', hue='Churn', data=df)
plt.title("Churn by DeviceProtection")
plt.xlabel("DeviceProtection")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='TechSupport', hue='Churn', data=df)
plt.title("Churn by TechSupport")
plt.xlabel("TechSupport")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='StreamingTV', hue='Churn', data=df)
plt.title("Churn by StreamingTV")
plt.xlabel("StreamingTV")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='StreamingMovies', hue='Churn', data=df)
plt.title("Churn by StreamingMovies")
plt.xlabel("StreamingMovies")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract")
plt.xlabel("Contract")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='PaperlessBilling', hue='Churn', data=df)
plt.title("Churn by PaperlessBilling")
plt.xlabel("PaperlessBilling")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title("Churn by PaymentMethod")
plt.xlabel("PaymentMethod")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(10,8))
sns.histplot(data=df,x='tenure', hue='Churn', kde=True, bins=20)
plt.title("Churn by tenure")
plt.xlabel("tenure")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(10,8))
sns.histplot(data=df,x='MonthlyCharges', hue='Churn', kde =True ,bins=20)
plt.title("Churn by MonthlyCharges")
plt.xlabel("MonthlyCharges")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(10,8))
sns.histplot(data=df, x='TotalCharges', hue='Churn', kde=True,bins=20)
plt.title("Churn by TotalCharges")
plt.xlabel("totalCharges")
plt.ylabel("COunt")
plt.legend(title = "Churn",loc='upper right', labels=['No Churn', 'Churn'])
plt.show()

le = LabelEncoder()
Label = df.select_dtypes(include=['object'])
df1 = df.copy()

for  i in Label:
    df1[i] = le.fit_transform(df1[i])

df1.shape

df1.dtypes

df1.head()

X = df1.iloc[:,1:20]
Y = df1.iloc[:,-1]

X.head()

mi_score1 = mutual_info_classif(X,Y)
mi_score1 = pd.Series(mi_score1)
mi_score1.index = X.columns
mi_score1.sort_values(ascending=True)

mi_score1.sort_values(ascending=False).plot.bar(figsize=(20,16))

X["Usage_Bill_Ratio"] = X["MonthlyCharges"] / X["TotalCharges"]

mi_score1 = mutual_info_classif(X,Y)
mi_score1 = pd.Series(mi_score1)
mi_score1.index = X.columns
mi_score1.sort_values(ascending=True)

mi_score1.sort_values(ascending=False).plot.bar(figsize=(20,16))

train_data,test_data,train_label,test_label = train_test_split(X,Y,test_size=0.3,random_state=0)

print("train_data :",train_data.shape)
print("train_label :",train_label.shape)
print("test_data :",test_data.shape)
print("test_label :",test_label.shape)

sc = StandardScaler()
train_data_sc = sc.fit_transform(train_data)
test_data_sc = sc.fit_transform(test_data)

print(train_data_sc)

pc = PCA()
train_data_sc_pc = pc.fit_transform(train_data_sc)
test_data_sc_pc = pc.fit_transform(test_data_sc)

explained_variance = pc.explained_variance_ratio_

print("Explained Varinace Rations :",explained_variance)

cumulative_variance = np.cumsum(explained_variance)

plt.plot(range(1,len(explained_variance) + 1), cumulative_variance,marker='o', linestyle = '-')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title('Scree Plot or Cumulative Explained Variance Plot')
plt.grid(True)
plt.show()

cumulative_variance = np.cumsum(explained_variance)
desired_variance = 0.95
num_components = np.argmax(cumulative_variance >= desired_variance) + 1
print(f"\nNumber of Components Selected: {num_components}")

pc = PCA(n_components=16)
train_data_sc_pc_select = pc.fit_transform(train_data_sc)
test_data_sc_pc_select = pc.fit_transform(test_data_sc)

explained_variance  = pc.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)

plt.plot(range(1,len(explained_variance) + 1), cumulative_variance,marker='o', linestyle = '-')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title('Scree Plot or Cumulative Explained Variance Plot')
plt.grid(True)
plt.show()

print("train_data :",train_data_sc_pc_select.shape)
print("test_data :",test_data_sc_pc_select.shape)

model_lr = LogisticRegression()

model_lr.fit(train_data_sc_pc_select, train_label)

y_pred = model_lr.predict(test_data_sc_pc_select)
y_pred

print("Accuracy_Score :", accuracy_score(y_pred, test_label))

confusion_matrix(y_pred,test_label)

print(classification_report(y_pred,test_label))

print("Cross _Val_Score Train Data : ", cross_val_score(model_lr,train_data_sc_pc_select,train_label, cv=5).mean())
print("Cross _Val_Score Test_Data : ", cross_val_score(model_lr,test_data_sc_pc_select,test_label,cv=5).mean())

from sklearn.model_selection import GridSearchCV

param_grid = {'C' : [0.001, 0.01, 0.1, 1,10, 100]}

model_lr = LogisticRegression()
grid_search = GridSearchCV(model_lr, param_grid, cv=5)
grid_search.fit(train_data_sc_pc_select,train_label)
best_params = grid_search.best_params_
print("Best Hyperparameters :", best_params)
best_score = grid_search.best_score_
print("best SCORE : ", best_score)

from sklearn.metrics._plot.roc_curve import auc

fpr, tpr, thresholds = roc_curve(test_label,y_pred)

roc_auc = auc (fpr,tpr)

plt.figure(figsize=(12,10))
plt.plot(fpr,tpr, color = "darkorange", lw=2, label ="ROC curve(area = %0.2f)" % roc_auc)
plt.plot([0,1],[0,1], color ='navy',lw=2, linestyle ='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc = 'lower right')
plt.show()         

model_rf = RandomForestClassifier()

model_rf.fit(train_data_sc_pc_select,train_label)

y_pred_2 = model_rf.predict(test_data_sc_pc_select)
y_pred_2

print("Train Data score : ",model_rf.score(train_data_sc_pc_select,train_label))
print("Test Data score : ",accuracy_score(y_pred_2,test_label))

print(classification_report(y_pred_2,test_label))

print("Cross_Val_Score Train Data : ",cross_val_score(model_rf,train_data_sc_pc_select,train_label,cv=5).mean())
print("Cross _Val_Score Test Data : ",cross_val_score(model_rf,test_data_sc_pc_select,test_label,cv=5).mean())

from sklearn.metrics._plot.roc_curve import auc

fpr, tpr, thresholds = roc_curve(test_label,y_pred_2)

roc_auc = auc (fpr,tpr)

plt.figure(figsize=(12,10))
plt.plot(fpr,tpr, color = "darkorange", lw=2, label ='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1], color ='navy',lw=2, linestyle ='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc = 'lower right')
plt.show()  

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators' : [50, 100, 200],
#     'max_depth' : [None, 5, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_sample_leaf' : [1, 2, 4]
# }

# rf_classifier = RandomForestClassifier(random_state=42)
# # rf_classifier = RandomForestClassifier(min_samples_leaf=5, random_state=42)


# grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)


# grid_search.fit(X, Y)
# best_params = grid_search.best_params_
# print("best hyperparameters :", best_params)

# best_score = grid_search.best_score_
# print("Best Score :", best_score)

model_rf = RandomForestClassifier(max_depth=10, min_samples_leaf= 2, min_samples_split=5, n_estimators=50)
model_rf.fit(train_data_sc_pc_select,train_label)
y_pred_2 = model_rf.predict(test_data_sc_pc_select)
print("Train Data Score : ",model_rf.score(train_data_sc_pc_select, train_label))
print("Test Data Accuracy Score : ",model_rf.score(test_data_sc_pc_select,test_label))

print("Cross _Val_Score Train Data : ",cross_val_score(model_rf,train_data_sc_pc_select,train_label,cv=5).mean())
print("Cross _Val_Score Test Data : ",cross_val_score(model_rf,test_data_sc_pc_select,test_label,cv=5).mean())

model_xg = xgb.XGBClassifier()

model_xg.fit(train_data_sc_pc_select,train_label)

y_pred_3 = model_xg.predict(test_data_sc_pc_select)

print("Train Data Score : ",model_xg.score(train_data_sc_pc_select, train_label))
print("Test Data Accuracy Score : ",accuracy_score(y_pred_3,test_label))

print("Cross _Val_Score Train Data : ",cross_val_score(model_xg,train_data_sc_pc_select,train_label,cv=5).mean())
print("Cross _Val_Score Test Data : ",cross_val_score(model_xg,test_data_sc_pc_select,test_label,cv=5).mean())

print(classification_report(y_pred_3,test_label))

from sklearn.metrics._plot.roc_curve import auc

fpr, tpr, thresholds = roc_curve(test_label,y_pred_3)

roc_auc = auc (fpr,tpr)

plt.figure(figsize=(12,10))
plt.plot(fpr,tpr, color = "darkorange", lw=2, label ='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1], color ='navy',lw=2, linestyle ='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc = 'lower right')
plt.show()  













