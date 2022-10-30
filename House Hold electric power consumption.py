#!/usr/bin/env python
# coding: utf-8

# ![Individual-household-electric-power-consumption-Data-Set--2.png](attachment:Individual-household-electric-power-consumption-Data-Set--2.png)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Read Data Set 
# 
# # Data Set Information:
# 
# '''This archive contains 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 (47 months).
# Notes:
# 1.(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3.
# 2.The dataset contains some missing values in the measurements (nearly 1,25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. For instance, the dataset shows missing values on April 28, 2007.
# 
# 
# Attribute Information:
# 
# 1.date: Date in format dd/mm/yyyy
# 2.time: time in format hh:mm:ss
# 3.global_active_power: household global minute-averaged active power (in kilowatt)
# 4.global_reactive_power: household global minute-averaged reactive power (in kilowatt)
# 5.voltage: minute-averaged voltage (in volt)
# 6.global_intensity: household global minute-averaged current intensity (in ampere)
# 7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
# 8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
# 9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.'''

# # Number of Records = 2075259

# In[2]:


data=data=pd.read_csv(r'C:\Users\anubh\Downloads\household_power_consumption\household_power_consumption.txt',delimiter=";",usecols=['Date','Time','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])


# # Take a sample data of 50K

# In[3]:


df=data.sample(n=50000)


# In[4]:


df.info()


# In[5]:


df.columns


# # Drop missing values

# In[6]:


df.dropna(inplace=True)


# # Convert data type for each columns

# In[7]:


df=df.astype({'Global_active_power':'float','Global_reactive_power':'float','Voltage':'float', 'Global_intensity':'float', 'Sub_metering_1':'float', 'Sub_metering_2':'float'})


# In[8]:


df['Date']=pd.to_datetime(df['Date'])


# In[9]:


df['Time']=pd.to_datetime(df['Time'])


# In[10]:


df.info()


# In[11]:


df.head()


# # Extract new column from Date and Time Column.
# 
# # Merge Sub_metering 1 ,2 & 3. Convert into Total Metering
# 
# # Later dropped unuseful conlumn`

# In[12]:


df['year']=df['Date'].dt.year


# In[13]:


df['month']=df['Date'].dt.month
df['days']=df['Date'].dt.day


# In[14]:


df['time']=df['Time'].dt.time


# In[15]:


df['hour']=df['Time'].dt.hour


# In[16]:


df['minutes']=df['Time'].dt.minute


# In[17]:


df.head()


# In[18]:


df['Total_metering']=df['Sub_metering_1']+df['Sub_metering_2']+df['Sub_metering_3']


# In[19]:


df.head()


# In[20]:


df.drop(['Date','Time','time','Sub_metering_1','Sub_metering_2','Sub_metering_3'],axis=1,inplace=True)


# # Final Data

# In[21]:


df.head()


# # EDA

# In[22]:


df.describe()


# In[23]:


df.corr()


# # Univariate analysis

# In[24]:


numerical_feature=[feature for feature in df.columns if df[feature].dtype!='O']

print('The numerical feature is {} and the feature are {}'.format(len(numerical_feature),numerical_feature))


# In[25]:


for feature in numerical_feature:
    sns.histplot(data=df,x=feature,kde=True,color='g')
    plt.show()


# # Bivariate analysis

# In[73]:


for feature in numerical_feature:
    sns.scatterplot(data=df,x=feature,y='Total_metering')
    plt.show()


# # Find outlier

# In[26]:


for feature in numerical_feature:
    sns.boxplot(data=df,x=feature,color='g')
    plt.show()


# # Multivariate analysis

# In[27]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# In[74]:


sns.pairplot(df)


# In[28]:


df.head()


# # Split data the train and test 
# 
# # applied Scaling (StandardScaler)

# In[29]:


`X=df.drop(['Total_metering'],axis=1)


# In[30]:


y=df['Total_metering']


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.30)


# In[33]:


X_train.shape


# In[34]:


y_train.shape


# In[35]:


X_test.shape


# In[36]:


y_test.shape


# In[37]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[38]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[39]:


X_train


# In[40]:


X_test


# # model1 LinearRegression

# In[41]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()


# In[42]:


lr.fit(X_train,y_train)


# In[43]:


lr_pred=lr.predict(X_test)


# In[44]:


lr_pred


# In[45]:


print(lr.coef_)


# In[46]:


print(lr.intercept_)


# # Assumption of Linear Regression 
# 
# # 1. there is linear realtionship between truth value and pridicted value 

# In[47]:


sns.scatterplot(y_test,lr_pred)


# # 2. if we calculate the residual(error) and plot. it should follow the normal distribution.

# In[48]:


residual=y_test-lr_pred
sns.distplot(residual,kde=True)


# # 3. if we plot on graph the pridicted value and residual. they should be follow unifrom distribution.

# In[49]:


sns.scatterplot(lr_pred,residual)


# In[50]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[51]:


print(mean_absolute_error(y_test,lr_pred))
print(mean_squared_error(y_test,lr_pred))
print(np.sqrt(mean_squared_error(y_test,lr_pred)))


# In[52]:


from sklearn.metrics import r2_score
score=r2_score(y_test,lr_pred)
print(score)


# In[53]:


# adjusted R square 
1-(1-score)*len(y_test)/(len(y_test)-X_test.shape[1]-1)


# # Model1 Linear Regression - Accuracy Score is 72 %

# In[54]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet

ridge=Ridge()
lasso=Lasso()
el_net=ElasticNet()


# # model2 Ridge Regression 

# In[55]:


ridge.fit(X_train,y_train)


# In[56]:


ridge_pred=ridge.predict(X_test)


# In[57]:


ridge.coef_


# In[58]:


ridge.intercept_


# # Assumption 
# 
# # 1. there is linear realtionship between truth value and pridicted value 
# # 2. if we calculate the residual(error) and plot. it should follow the normal distribution
# # 3. if we plot on graph the pridicted value and residual. they should be follow unifrom distribution.

# In[59]:


sns.scatterplot(y_test,ridge_pred)


# In[60]:


ridge_residual=y_test-ridge_pred
sns.distplot(residual,kde=True)


# In[61]:


sns.scatterplot(ridge_pred,ridge_residual)


# In[62]:


print(mean_absolute_error(y_test,ridge_pred))
print(mean_squared_error(y_test,ridge_pred))
print(np.sqrt(mean_squared_error(y_test,ridge_pred)))


# In[63]:


r2_score(y_test,ridge_pred)


# # model2 Ridge Regression - score is 72%

# # Model3 Lasso 

# In[64]:


lasso.fit(X_train,y_train)


# In[65]:


lasso_pred=lasso.predict(X_test)


# # Assumption 
# 
# # 1. there is linear realtionship between truth value and pridicted value 
# # 2. if we calculate the residual(error) and plot. it should follow the normal distribution
# # 3. if we plot on graph the pridicted value and residual. they should be follow unifrom distribution.

# In[77]:


sns.scatterplot(y_test,lasso_pred)


# In[79]:


lasso_residual=y_test-lasso_pred
sns.distplot(lasso_residual,kde=True)


# In[80]:


sns.scatterplot(lasso_pred,lasso_residual)


# In[66]:


r2_score(y_test,lasso_pred)


# # Model3 accuracy score is 70%

# # Model 4 ElasticNet

# In[67]:


el_net.fit(X_train,y_train)


# In[68]:


el_net_pred=el_net.predict(X_test)


# In[69]:





# # Assumption 
# 
# # 1. there is linear realtionship between truth value and pridicted value 
# # 2. if we calculate the residual(error) and plot. it should follow the normal distribution
# # 3. if we plot on graph the pridicted value and residual. they should be follow unifrom distribution.

# In[81]:


sns.scatterplot(y_test,el_net_pred)


# In[83]:


el_net_residual=y_test-el_net_pred
sns.distplot(el_net_residual,kde=True)


# In[84]:


sns.scatterplot(el_net_pred,el_net_residual)


# In[85]:


r2_score(y_test,el_net_pred)


# # Model4 ElasticNet accuracy score is 67 %

# # With Data set the Linear & Ridge Regression perform better accuracy

# In[ ]:




