#Importing Packages

import pandas as pd #for data processing
import numpy as np #for arrays
import matplotlib.pyplot as plt #for visualization
import seaborn as sb #also for visualization
from termcolor import colored as cl #for text customization

from sklearn import svm #support vector machine

from sklearn.model_selection import train_test_split #for data split

from sklearn.linear_model import LinearRegression #OLS algorithm

from sklearn.metrics import r2_score as r2 #also evaluation metrics
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor #random forest regression

sb.set_style('whitegrid') #plot style
plt.rcParams['figure.figsize'] = (20,10) #plot size

#importing data

df = pd.read_csv('house.csv')
# df.set_index('Id', inplace = True)

print(df.head(5))

print(df.describe())#statistical view of the data

print(cl(df.dtypes, attrs = ['bold']))#checking dataset

#differentitaing categorical variables(object data type), int variables(int datatype), float variables(float datatype)
obj = (df.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))
 
int_ = (df.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))
 
fl = (df.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))

# Excluding object columns from correlation calculation
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_cols].corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 6))
sb.heatmap(correlation_matrix, cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
# plt.show()

# barplot
unique_values = []
for col in object_cols:
    unique_values.append(df[col].nunique())
plt.figure(figsize=(10, 6))
plt.title('No. Unique values of categorical features')
plt.xticks(rotation=90)
sb.barplot(x=object_cols, y=unique_values)
# plt.show()

plt.figure(figsize=(18, 36))
plt.suptitle('Categorical Features: Distribution', y=0.92, fontsize=16)
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = df[col].value_counts()
    ax = plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sb.barplot(x=list(y.index), y=y)
    index += 1

plt.show()

#Data cleaning
df.drop(['Id'], axis=1,	inplace=True)

#Replacing SalePrice empty values with their mean values to make the data more symmetric
df['SalePrice'] = df['SalePrice'].fillna(df['SalePrice'].mean())

#drop records with null values
new_df = df.dropna()

#check if there are still null values
print(new_df.isnull().sum())

#for label categorical features
s = (new_df.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', len(object_cols))

OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_df[object_cols]))
OH_cols.index = new_df.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_df.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

#splitting dataset into training and testing
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']
 
# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

#svm
model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))

#random forest regression
model_RFR = RandomForestRegressor(n_estimators = 10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))

#linear regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid) 

print(mean_absolute_percentage_error(Y_valid, Y_pred))

# Calculate R-squared
r2 = r2(Y_valid, Y_pred)
print("R-squared: ", r2)

