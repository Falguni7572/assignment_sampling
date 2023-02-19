import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from numpy import where
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


data = pd.read_csv("Creditcard_data.csv")
# print(data)

x = data.drop("Class", axis='columns')
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


undersample = NearMiss(version=2, n_neighbors=3)
# transform the dataset
Xu2, yu2 = undersample.fit_resample(X_train, y_train)

lr4 = LogisticRegression()
lr4.fit(Xu2, yu2.ravel())
predictions = lr4.predict(X_test)
  
# print classification report
# print(classification_report(y_test, predictions))

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

lr1 = LogisticRegression()
lr1.fit(X_train_res, y_train_res.ravel())
predictions = lr1.predict(X_test)
  
# print classification report
# print(classification_report(y_test, predictions))



###########################################################
# therefore we use oversampling SMOTE as there 
# is a possibility that undersampling might overfit the data

sm = SMOTE(random_state = 2)
X, y = sm.fit_resample(x, y.ravel())

X_train_res, X_test, y_train_res, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


aux1 = X_train_res
aux1['class'] = y_train_res
df = aux1

###############################################################
# simple random sampling
sampled_df = df.sample(frac=0.7, random_state=42, replace = True)
# print(sampled_df)

###############################################################
# simple random sampling without replacement
samp_rep_df = df.sample(frac=0.7, random_state=42, replace = False)
# print(samp_rep_df)

###############################################################
# systematic sampling
i = 3
systematic_df = df.iloc[::i]
# print(systematic_df)

##############################################################
# stratified sampling 
n = int((1.96*1.96 * 0.5*0.5)/((0.05)**2))

strata = df.groupby('class')
# sample 2 rows from each stratum
stratified_df = strata.apply(lambda x: x.sample(n))
# print(stratified_df)

#################################################################
# cluster sampling

kmeans = KMeans(n_clusters=7, random_state=42).fit(df)
cluster_assignments = kmeans.labels_

# Select the clusters you want to include in the sample
selected_clusters = [2,4,5]

cluster_series = pd.Series(cluster_assignments)

df["cluster"] = cluster_series

df_cluster_sample = pd.DataFrame()
for i in selected_clusters:
    aux3 = df.loc[df["cluster"]==i]
    df_cluster_sample = df_cluster_sample.append(aux3, ignore_index = True)
    
df = df.drop('cluster', axis="columns")
df_cluster_sample = df_cluster_sample.drop('cluster', axis="columns")

################################################################
# now applying models

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': XGBClassifier()
}

###############################################################
# simple random sampling with replacement
X_train_now = sampled_df.drop("class", axis='columns')
y_train_now = sampled_df['class']
result_simple_rep = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_simple_rep.append(accuracy)

##############################################################
# simple random sampling without replacement

X_train_now = samp_rep_df.drop("class", axis='columns')
y_train_now = samp_rep_df['class']
result_samp_rep_df = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_samp_rep_df.append(accuracy)

###############################################################
# systematic sampling

X_train_now = systematic_df.drop("class", axis='columns')
y_train_now = systematic_df['class']
result_systematic_df = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_systematic_df.append(accuracy)

##############################################################
# stratified sampling
X_train_now = stratified_df.drop("class", axis='columns')
y_train_now = stratified_df['class']
result_stratified_df = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_stratified_df.append(accuracy)

#################################################################
# cluster sampling
X_train_now = df_cluster_sample.drop("class", axis='columns')
y_train_now = df_cluster_sample['class']
result_df_cluster_sample = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_df_cluster_sample.append(accuracy)

##################################################################
# display the table

final_result = pd.DataFrame()
final_result['simple random with replacement'] = result_simple_rep
final_result['simple random without replacement'] = result_samp_rep_df
final_result['systematic sampling'] = result_systematic_df
final_result['stratified sampling'] = result_stratified_df
final_result['cluster sampling'] = result_df_cluster_sample

rows = {
    0 : 'Logistic Regression',
    1: 'Random Forest',
    2: 'Support Vector Machine',
    3 :'K-Nearest Neighbors',
    4 : 'XGBoost'
}

table_n = final_result.rename(index=rows)

print("final output: ")
print(table_n)