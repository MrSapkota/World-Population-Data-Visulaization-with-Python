#!/usr/bin/env python
# coding: utf-8

# # Data loading and Preprocessing

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Loading data files
hw= pd.read_csv('height_weight_data.csv')
iq= pd.read_csv('iq.csv')
life_expectancy= pd.read_csv('life_expectancy.csv')
population_density= pd.read_csv('population_density.csv')
qof= pd.read_csv('quality_of_life.csv')


# In[ ]:


hw.shape


# In[ ]:


iq.shape


# In[ ]:


life_expectancy.shape


# In[ ]:


population_density.shape


# In[ ]:


qof.shape


# # # view first Few rows

# In[ ]:


hw.head(2)


# In[ ]:


iq.head(2)


# In[ ]:


life_expectancy.head(2)


# In[ ]:


population_density.head(2)


# In[ ]:


qof.head(2)


# ## Simple statastical evaluation of data

# In[ ]:


#describe height_weight data
hw.describe()


# In[ ]:


#describe iq data
iq.describe()


# In[ ]:


#describe life_expectancy data
life_expectancy.describe()


# In[ ]:


#describe population_density data
population_density.describe()


# In[ ]:


#describe Quality of life data
qof.describe()


# ## Merging the dataset

# In[ ]:


df_1 = pd.merge(life_expectancy,qof, how = 'left')

df_2 = pd.merge(df_1, iq, how = 'left')

df_3 = pd.merge(df_2, population_density, how = 'left')

world_pop_data = pd.merge(df_3, hw, how = 'left')


# In[ ]:


world_pop_data.shape


# In[ ]:


world_pop_data.head()


# STATISTICAL DATA EVALUATION OF WORLD POPULATION DATA

# In[ ]:


#Statistical Evaluation of World_pop_data
world_pop_data.describe()


# In[ ]:


world_pop_data.info()


# ## checking missing values

# In[ ]:


world_pop_data.isnull().sum()


# ## Handling missing values

# In[ ]:


world_pop_data.fillna(world_pop_data.mean(),inplace=True)


# In[ ]:


world_pop_data.isnull().sum()


# In[ ]:


#Encoding object(string) type to int type.
wpd = world_pop_data.copy()
labelencoder = LabelEncoder()
wpd['country'] = labelencoder.fit_transform(wpd['country'])


# In[ ]:


#fill the missing values in education_expenditure_per_inhabitant and population column with "mode" strategy
wpd['education_expenditure_per_inhabitant'].fillna(wpd['education_expenditure_per_inhabitant'].mode()[0], inplace=True)
wpd['population'].fillna(wpd['population'].mode()[0], inplace=True)


# In[ ]:


wpd.isnull().sum()


# ## Type Conversion
# 
# Here we are converting the data type of the variable education_expenditure_per_inhabitant and population from object to float type.

# In[ ]:


import re

def review_cleaning(text):
    
    # Removing unuseful character.
    text = re.sub(r',', '', text)
    return text

wpd['population'] = wpd['population'].apply(review_cleaning)


# In[ ]:


wpd['population']  = [i.split(' ')[0] for i in wpd['population'] ]


# In[ ]:


#Converting population column from object to float type
wpd['population']  = wpd['population'].astype('float64')


# In[ ]:


wpd['education_expenditure_per_inhabitant'] = wpd['education_expenditure_per_inhabitant'].apply(review_cleaning)


# ## CHECKING FOR OUTLIERS

# In[ ]:


# Calculate the first and third quartiles
Q1 = wpd.quantile(0.25)
Q3 = wpd.quantile(0.75)

# Calculate the IQR for each column
IQR = Q3 - Q1

# Define a threshold for outlier detection (e.g., 1.5 times IQR)
threshold = 1.5*IQR

# Find and print the columns with outliers
outliers = ((wpd < (Q1 - threshold)) | (wpd > (Q3 + threshold))).any(axis=0)
print("Indices of outliers:\n",outliers)


# ## HANDLING OUTLIERS

# In[ ]:


#Handling outliers in female_life_expectancy Column
Q1 = wpd['female_life_expectancy'].quantile(0.25)
Q3 = wpd['female_life_expectancy'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['female_life_expectancy'].median()
wpd['fle']=np.where((wpd['female_life_expectancy']< (Q1 - threshold))| (wpd['female_life_expectancy']> (Q3 + threshold)), median_value, wpd['male_life_expectancy'])


# In[ ]:


#Handling outliers in female_life_expectancy Column
Q1 = wpd['female_life_expectancy'].quantile(0.25)
Q3 = wpd['female_life_expectancy'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['female_life_expectancy'].median()
wpd['fle']=np.where((wpd['female_life_expectancy']< (Q1 - threshold))| (wpd['female_life_expectancy']> (Q3 + threshold)),median_value, wpd['female_life_expectancy'])


# In[ ]:


#Handling outliers in death_rate Column
Q1 = wpd['death_rate'].quantile(0.25)
Q3 = wpd['death_rate'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['death_rate'].median()
wpd['Death_rate']=np.where((wpd['death_rate']< (Q1 - threshold))| (wpd['death_rate']> (Q3 + threshold)), median_value, wpd['death_rate'])


# In[ ]:


#Handling outliers in safety Column
Q1 = wpd['safety'].quantile(0.25)
Q3 = wpd['safety'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['safety'].median()
wpd['Safety']=np.where((wpd['safety']< (Q1 - threshold))| (wpd['safety']> (Q3 + threshold)), median_value, wpd['safety'])


# In[ ]:


#Handling outliers in health Column
Q1 = wpd['health'].quantile(0.25)
Q3 = wpd['health'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['health'].median()
wpd['Health']=np.where((wpd['health']< (Q1 - threshold))| (wpd['health']> (Q3 + threshold)), median_value, wpd['health'])


# In[ ]:


#Handling outliers in climate Column
Q1 = wpd['climate'].quantile(0.25)
Q3 = wpd['climate'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['climate'].median()
wpd['Climate']=np.where((wpd['climate']< (Q1 - threshold))| (wpd['climate']> (Q3 + threshold)), median_value, wpd['climate'])


# In[ ]:


#Handling outliers in iq Column
Q1 = wpd['iq'].quantile(0.25)
Q3 = wpd['iq'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['iq'].median()
wpd['IQ']=np.where((wpd['iq']< (Q1 - threshold))| (wpd['iq']> (Q3 + threshold)), median_value, wpd['iq'])


# In[ ]:


#Handling outliers in popularity Column
Q1 = wpd['popularity'].quantile(0.25)
Q3 = wpd['popularity'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['popularity'].median()
wpd['Popularity']=np.where((wpd['popularity']< (Q1 - threshold))| (wpd['popularity']> (Q3 + threshold)), median_value, wpd['popularity'])


# In[ ]:


#Handling outliers in birth_rate Column
Q1 = wpd['birth_rate'].quantile(0.25)
Q3 = wpd['birth_rate'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['birth_rate'].median()
wpd['Birth_rate']=np.where((wpd['birth_rate']< (Q1 - threshold))| (wpd['birth_rate']> (Q3 + threshold)), median_value, wpd['birth_rate'])


# In[ ]:


#Handling outliers in population Column
Q1 = wpd['population'].quantile(0.25)
Q3 = wpd['population'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['population'].median()
wpd['Population']=np.where((wpd['population']< (Q1 - threshold))| (wpd['population']> (Q3 + threshold)), median_value, wpd['population'])


# In[ ]:


#Handling outliers in pop_per_km_sq Column
Q1 = wpd['pop_per_km_sq'].quantile(0.25)
Q3 = wpd['pop_per_km_sq'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['pop_per_km_sq'].median()
wpd['Pop_density']=np.where((wpd['pop_per_km_sq']< (Q1 - threshold))| (wpd['pop_per_km_sq']> (Q3 + threshold)), median_value, wpd['pop_per_km_sq'])


# In[ ]:


#Handling outliers in area Column
Q1 = wpd['area'].quantile(0.25)
Q3 = wpd['area'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['area'].median()
wpd['Area']=np.where((wpd['area']< (Q1 - threshold))| (wpd['area']> (Q3 + threshold)), median_value, wpd['area'])


# In[ ]:


#Handling outliers in male_height Column
Q1 = wpd['male_height'].quantile(0.25)
Q3 = wpd['male_height'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['male_height'].median()
wpd['Male_height']=np.where((wpd['male_height']< (Q1 - threshold))| (wpd['male_height']> (Q3 + threshold)), median_value, wpd['male_height'])


# In[ ]:


#Handling outliers in female_height Column
Q1 = wpd['female_weight'].quantile(0.25)
Q3 = wpd['female_weight'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['female_weight'].median()
wpd['Female_weight']=np.where((wpd['female_weight']< (Q1 - threshold))| (wpd['female_weight']> (Q3 + threshold)), median_value, wpd['female_weight'])


# In[ ]:


#Handling outliers in female_bmi Column
Q1 = wpd['female_bmi'].quantile(0.25)
Q3 = wpd['female_bmi'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['female_bmi'].median()
wpd['Female_bmi']=np.where((wpd['female_bmi']< (Q1 - threshold))| (wpd['female_bmi']> (Q3 + threshold)), median_value, wpd['female_bmi'])


# In[ ]:


#Handling outliers in male_bmi Column
Q1 = wpd['male_bmi'].quantile(0.25)
Q3 = wpd['male_bmi'].quantile(0.75)
IQR=Q3-Q1
threshold = 1.5*IQR
median_value = wpd['male_bmi'].median()
wpd['Male_bmi']=np.where((wpd['male_bmi']< (Q1 - threshold))| (wpd['male_bmi']> (Q3 + threshold)), median_value, wpd['male_bmi'])


# In[ ]:


#Dropping columns with outliers
wpd=wpd.drop(columns=["female_weight","female_bmi","male_bmi","male_height","pop_per_km_sq","population","area","education_expenditure_per_inhabitant","iq","popularity","climate","safety","health","male_life_expectancy","female_life_expectancy","birth_rate","death_rate"],axis=1)
 


# In[ ]:


# Check the data statistics again 
wpd.describe()


# ## Selecting target variable

# In[ ]:


##combining target
    mean = (wpd['mle'] + (wpd['fle']) / 2
    print(mean)


# IDENTIFYING THE MACHINE LEARNING MODEL TASK

# #IDENTIFYING RELEVANT VARIABLES

# In[ ]:


correlation_matrix = wpd.corr()
cor_with_life_expectancy = correlation_matrix['life_expectancy'].abs().sort_values(ascending=False)
print(cor_with_life_expectancy)


# # Visualization using Scatter Plot

# In[ ]:


wpd_dummy=wpd.copy()
wpd_dummy.drop(columns = ['mle','fle'],inplace=True)
Features=wpd_dummy.drop(columns='life_expectancy')
fig1 = plt.figure(figsize=(12,12))
#sns.set(style='whitegrid', palette='Set1')
plotnumber = 1

for feature in Features:
    ax = plt.subplot(6,4, plotnumber)
    sns.scatterplot(data=wpd_dummy,x= feature,y="life_expectancy")
    plt.xlabel(feature, size = 12)
    plt.title(f'Life Expectancy by {feature}', size = 10)
    #plt.xticks(size = 15, rotation = 90)
    #plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0., fontsize=16)
    ax.set_ylabel('', fontsize = 10)
    #plt.yticks(size = 18)
    plotnumber += 1
    plt.tight_layout();


# # SETTING PREDICTORS AND TARGET VARIABLE

# In[ ]:


X=wpd.drop(columns=["country","male_weight","Male_height","Edu_expenditure","female_height","daily_max_temp","costs","Female_weight","Area","Climate","Pop_density","Population","Popularity","Death_rate","life_expectancy"],axis=1)
y=wpd["life_expectancy"]


# In[ ]:


#Converting y data into 1D array
y=np.array(y)


# In[ ]:


#Converting 1D array into 2D array
y=y.reshape(-1,1)


# Methods to use
# For the analysis, you need to analyse the data using at least three different methods.
# You’ll need to use
# • Neural Networks, and
# • Support Vector Machine
# as two of the methods. Do not use linear and logistic regression other than as a comparison method. The other methods taught in this subject which you can use are:
# • Ridge Lasso regression,
# • Lasso regression, and
# • Naive Bayes

# ## Lasso Regression

# Fine-tuning Hyperparameters (if necessary):
# You can perform cross-validation to find the optimal value of the hyperparameter alpha.

# In[ ]:


# import necessary packages

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# create train-test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10)
# transform the training data
scalerDeath = StandardScaler()
x_train_scaled = scalerDeath.fit_transform(X_train)


# In[ ]:


#Splitting data into training and test data
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=10)


# In[ ]:


#Standardising train and test data
sx=StandardScaler()
sy=StandardScaler()
Xtrain_scaled=sx.fit_transform(Xtrain)
Xtest_scaled=sx.transform(Xtest)
ytrain_scaled=sy.fit_transform(ytrain)
ytest_scaled=sy.transform(ytest)


# In[ ]:


print('mean: ', np.mean(x_train_scaled, axis=10))
print('sd: ', np.std(x_train_scaled, axis=10))


# In[ ]:


# transform the test data
x_test_scaled = scalerDeath.transform(X_test)


# In[ ]:


print('mean: ', np.mean(x_test_scaled, axis=0))
print('sd: ', np.std(x_test_scaled, axis=0))


# In[ ]:


#Training a  Lasso regression
alphas = np.arange(0.05, 4, 0.05)
test_error = []
for alpha in alphas:
lasso_reg = Lasso(alpha=alpha, random_state=42, max_iter = 1e4)
lasso_reg.fit(x_train_scaled, y_train)
ypred = lasso_reg.predict(x_test_scaled)
test_error.append(mean_squared_error(y_test, ypred))
indA = np.argmin(test_error)
print('Minimum test error: ', test_error[indA], ' at alpha equals:',alphas[indA])
plt.plot(alphas, test_error)
plt.show()


# In[ ]:


# validation
from sklearn.model_selection import KFold
for alpha in alphas:
kfold_err = []
kf = KFold(n_splits=5, shuffle=True, random_state = 42)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
# Now, normalise the data as above
# train Ridge with normalised training data
# prediction and calculate loss
## End Inner Loop
# Calculate cv error


# In[ ]:


lasso_reg2 = Lasso(alpha=2.2, random_state=42)
lasso_reg2.fit(x_train_scaled, y_train)
lasso_reg2.coef_


# In[ ]:




