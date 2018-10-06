# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:02:16 2018

@author: Sunny Parihar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('winequality-red.csv')
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.io import output_notebook
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import sklearn
import sklearn.metrics
from sklearn import ensemble
from sklearn import linear_model
import warnings
dataset.head(n=5)
dataset = pd.read_csv('winequality-red.csv',sep=';')
#print("Shape of Red Wine dataset: {s}").format(s = dataset.shape)
#print("Column headers/names: {s}").format(s = list(dataset))
dataset.info()
dataset.describe()
df = dataset.describe()
#Dealing with missing values of dataset.
dataset.isnull().sum()
#This dataset doesn't have any missing values.
#Rename the columns by removing space.
dataset.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'},inplace=True)
dataset.head(5)
#Check the unique values of quality column.
dataset['quality'].unique()
#counting the value of each quality.
dataset.quality.value_counts().sort_index()
sns.countplot(x='quality',data=dataset)
conditions=[(dataset['quality']>=7),
            (dataset['quality']<=4)
        ]
rating =['good','bad']
dataset['rating'] = np.select(conditions,rating,default='average')
dataset.rating.value_counts()
dataset.groupby('rating').mean()
#Correlation between target values and predictor values.
correlation = dataset.corr()
plt.figure(figsize=(12, 5))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
correlation['quality'].sort_values(ascending=False)
#Analysis the alcohol percentage with the wine quality.
bx = sns.boxplot(x="quality", y='alcohol', data = dataset)
bx.set(xlabel='Wine Quality', ylabel='Alcohol Percent', title='Alcohol percent in different wine quality types')
#Analysis of sulphates in wine ratings.
sx = sns.boxplot(x='rating',y='sulphates',data=dataset)
sx.set(xlabel='Wine Quality',ylabel='Suplate Amount',title='Relation between Sulphate with rating of wine')
#Analysis of citirc acid and wine ratings.
cx = sns.violinplot(x='rating',y='citric_acid',data=dataset)
cx.set(xlabel='Wine Ratings', ylabel='Citric Acid', title='Citric_acid in different types of Wine ratings')
#Analysis of fixed acidity and wine ratings.
fx = sns.boxplot(x="rating", y='fixed_acidity', data = dataset)
fx.set(xlabel='Wine Ratings', ylabel='Fixed Acidity', title='Fixed Acidity in different types of Wine ratings')
#Analysis of pH and wine ratings.
px = sns.swarmplot(x="rating", y="pH", data = dataset);
px.set(xlabel='Wine Ratings', ylabel='pH', title='pH in different types of Wine ratings')
sns.lmplot(x = "alcohol", y = "residual_sugar", col = "rating", data = dataset)
y,X = dmatrices('quality ~ alcohol', data=dataset, return_type='dataframe')
print("X:", type(X))
print(X.columns)
model=smf.OLS(y, X)
result=model.fit()
result.summary()
model = smf.OLS.from_formula('quality ~ alcohol', data = dataset)
results = model.fit()
print(results.params)
#Classification using stats model.
dataset['rate_code'] = (dataset['quality'] > 4).astype(np.float32)

y, X = dmatrices('rate_code ~ alcohol', data = dataset)
sns.distplot(X[y[:,0] > 0, 1])
sns.distplot(X[y[:,0] == 0, 1])


model = smf.Logit(y, X)
result = model.fit()
result.summary2()


yhat = result.predict(X)
sns.distplot(yhat[y[:,0] > 0])
sns.distplot(yhat[y[:,0] == 0])


yhat = result.predict(X) > 0.955
print(sklearn.metrics.classification_report(y, yhat))

#Classification using sklrean logistic regression.
model = sklearn.linear_model.LogisticRegression()
y,X = dmatrices('rate_code ~ alcohol + sulphates + citric_acid + fixed_acidity', data = dataset)
model.fit(X, y)
yhat = model.predict(X)
print(sklearn.metrics.classification_report(y, yhat))
#Using Random Forest Classifier.
    
y, X = dmatrices('rate_code ~ alcohol', data = dataset)
model = sklearn.ensemble.RandomForestClassifier()
model.fit(X, y)
yhat = model.predict(X)
print(sklearn.metrics.classification_report(y, yhat))



