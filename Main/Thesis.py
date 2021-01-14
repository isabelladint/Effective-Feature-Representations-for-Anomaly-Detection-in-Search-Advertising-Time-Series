#!/usr/bin/env python
# coding: utf-8

# In[20]:


#basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import RepeatedKFold, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from functools import reduce


# # 2. File with selected features, visualizations

# In[22]:


test2 = pd.read_csv("test2.csv", index_col=0, parse_dates=['date'])


# In[23]:


len(test2["campaignId"].unique())


# In[24]:


test2.info()


# In[25]:


test2.groupby(['campaignId','date'])['impressions'].sum().head(59).plot()
plt.xticks(rotation=45)


# # 3. Apply inactive campaign drop function

# In[26]:


cutoff = 30
campaign_counts = test2["campaignId"].value_counts()
active_campaigns = campaign_counts[campaign_counts >= cutoff]
test3 = test2[test2["campaignId"].isin(active_campaigns.index.values)]


# In[27]:


test3


# In[28]:


len(test3["campaignId"].unique())


# # 4. Impute the 30-58 day campaigns

# #### These were imputed with zeros since the shortest comparable period is 30 days, under which campaignId related observations were dropped, and for those which are in the range of 30-58 day length there is some missing data in order for them to be comparable to the longest active campaignId observations.

# In[29]:


# create a fresh dataframe copy
test4 = test3.copy(deep=True)

# retrieve all distinct campignIds
distinct_campaigns = test4["campaignId"].unique()

# create a desired range of dates
idx = pd.date_range(start='8/1/2020', end='9/29/2020', freq="D").tolist()

# for each unique campaign
for campaignId in distinct_campaigns:
    data = test4[test4["campaignId"] == campaignId]
   
   # for each required date
    for new_date in idx:
        exists = False
       
       # check if the date already exists in the data
        for existing_date in data["date"]:
            if existing_date == new_date:
                exists = True
                break
       
       # if not, insert a new row in the original dataset
        if not exists:
            row = data.iloc[0].copy(deep=True)
            row["date"] = new_date
            row["impressions"] = row["clicks"] = row["cost"] = 0
            empty_row = pd.DataFrame([row], columns=test4.keys().values)
            test4 = test4.append(empty_row)


# In[30]:


test4


# In[31]:


len(test4["campaignId"].unique())


# In[32]:


# check of the output - equal lengths of each campaignId 
print(test4[test4['campaignId']==7869105302]["date"].max()-test4[test4['campaignId']==7869105302]["date"].min())
print(test4[test4['campaignId']==7861968652]["date"].max()-test4[test4['campaignId']==7861968652]["date"].min())
print(test4[test4['campaignId']==10626807542]["date"].max()-test4[test4['campaignId']==10626807542]["date"].min())


# # 5. Apply portfolio separation function

# In[33]:


# separating the portfolio budget based on the number of campaigns under a googleadsid using it
def portfolio(row):
    portfolio = test4[test4["googleAdsId"] == row["googleAdsId"]]["campaignId"].nunique() 
    return row["campaignBudget"] / portfolio


# In[34]:


test4["portfolioBudget"] = test4.apply(lambda x: portfolio(x), axis=1)


# In[35]:


test4 = test4.drop("campaignBudget",axis=1)
test4


# In[36]:


len(test4.campaignId.unique())


# # 6. Further EDA and visualization

# Distributions via bar/histograms, and pairwise scatterplots

# In[37]:


from scipy import stats
pbiqr = stats.iqr(test4['portfolioBudget'], interpolation='midpoint')
pbq3 = np.percentile(test4['portfolioBudget'],75,interpolation='midpoint')
(pbiqr*1.5)+pbq3


# In[38]:


sns.set_theme(style="whitegrid")
sns.boxplot(x=test4.portfolioBudget)


# In[39]:


test4[test4['portfolioBudget']>3.75]['campaignId'].unique()


# In[40]:


sns.set_theme(style="whitegrid")
sns.boxplot(x=test4.portfolioBudget)


# In[41]:


test4[test4['portfolioBudget']>3]['campaignId'].unique()


# In[ ]:





# In[42]:


ciqr = stats.iqr(test4['cost'], interpolation='midpoint')
cq3 = np.percentile(test4['cost'],75,interpolation='midpoint')
(ciqr*1.5)+cq3


# In[43]:


sns.set_theme(style="whitegrid")
sns.boxplot(x=test4.cost)


# In[44]:


len(test4[test4['cost']>=3.75]['campaignId'].unique())


# In[45]:


sns.set_theme(style="whitegrid")
sns.boxplot(x=test4.clicks)


# In[ ]:





# In[ ]:





# In[46]:


# autocorrelation plot for the first campaign

from statsmodels.graphics import tsaplots

for cid in test4.campaignId.unique()[:1]:
    tsaplots.plot_acf(test4['clicks'], lags=40, title="Clicks")
    tsaplots.plot_acf(test4['impressions'], lags=40, title="Impressions")
    tsaplots.plot_acf(test4['cost'], lags=40, title="Cost")
    tsaplots.plot_acf(test4['portfolioBudget'], lags=40, title="Individual Budget")
    plt.show()


# In[47]:


sns.set_style("ticks")
sns.pairplot(test4, hue="biddingStrategyType")


# In[48]:


for i in test4['campaignId'].unique()[:1]:
    sns.lmplot(x="date", y="impressions", data=test4, fit_reg=False)
plt.show()


# # 7. Groupby and check for NaNs

# In[51]:


# if strategy same within same date, group by date in a new dataset
clean_grouped = test4.groupby(["date","campaignId","biddingStrategyType"]).agg({"impressions":"sum","clicks":"sum","cost":"sum","portfolioBudget":"sum"})
clean_grouped.head(20)


# In[52]:


# determining how many nans the dataset has
nans = len(clean_grouped) - clean_grouped.count()
nans


# # 8. Reset index and overview

# In[53]:


clean = clean_grouped.reset_index()


# In[54]:


clean['portfolioBudget'] = round(clean['portfolioBudget'],2)


# In[55]:


# checking out the dataframe
clean.head()


# In[56]:


clean.tail(2)


# In[57]:


clean.info()


# In[58]:


sns.set_style("ticks")
sns.pairplot(clean, hue='biddingStrategyType', diag_kind = 'kde',kind = 'scatter',palette = 'husl')


# # 9. Make numeric values where/if needed

# In[59]:


clean[["campaignId","impressions","clicks","cost","portfolioBudget"]].apply(pd.to_numeric, errors='coerce')
clean.info()


# # 10. Typecasting from category type is faster than from object

# In[60]:


clean["biddingStrategyType"] = clean["biddingStrategyType"].astype('category')
clean.info()


# In[61]:


# checking out the outcome distribution
# significantly more strategies of one class than the other two: 25071, 3966, 305
print(clean["biddingStrategyType"].value_counts())
print()
print("Number of categories in the outcome feature:", clean["biddingStrategyType"].value_counts().count())


# In[62]:


# checking for nas
nans = len(clean) - clean.count()
nans


# # 11. Visualizing categorical data

# In[63]:


# https://www.datacamp.com/community/tutorials/categorical-data
import seaborn as sns
bidtype_count = clean["biddingStrategyType"].value_counts()
sns.set(style="darkgrid")
sns.barplot(bidtype_count.index, bidtype_count.values, alpha=0.9)
plt.title('Frequency Distribution of Bidding Strategy Types', fontsize=13)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Bidding Strategy Type', fontsize=13)
plt.show()


# In[64]:


plots = clean.copy(deep=True)


# In[65]:


# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html

#ungrouped value distribution
fig, axs = plt.subplots(2, 2, figsize=(10,6))
axs[0, 0].plot(plots["date"], plots["impressions"])
axs[0, 0].set_title('Impressions over time')
axs[0, 1].plot(plots["date"], plots["clicks"], 'tab:orange')
axs[0, 1].set_title('Clicks over time')
axs[1, 0].plot(plots["date"], plots["cost"], 'tab:green')
axs[1, 0].set_title('Cost over time')
axs[1, 1].plot(plots["date"], plots["portfolioBudget"], 'tab:red')
axs[1, 1].set_title('Portfolio Budget over time')

for ax in axs.flat:
    ax.set(xlabel='Date')
    ax.tick_params(axis = 'x', labelrotation = 90)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()


# In[66]:


# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html

#ungrouped value distribution
fig, axs = plt.subplots(4, sharex=True, sharey=True, figsize=(10,6))
fig.suptitle('Sharing both axes')
axs[0].plot(clean["date"], clean["impressions"])
axs[1].plot(clean["date"], clean["clicks"], 'o')
axs[2].plot(clean["date"], clean["cost"], '+')
axs[3].plot(clean["date"], clean["portfolioBudget"], '-')

for axs in axs.flat:
    axs.tick_params(axis="x", labelrotation = 90)


# In[67]:


# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html

#ungrouped value distribution
fig, axs = plt.subplots(2, 2, figsize=(14,6))

axs[0, 0].plot(clean["date"], clean["impressions"])
axs[0, 0].set_title("Impressions")
axs[1, 0].plot(clean["date"], clean["clicks"])
axs[1, 0].set_title("Clicks")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(clean["date"], clean["cost"])
axs[0, 1].set_title("Cost")
axs[1, 1].plot(clean["date"], clean["portfolioBudget"])
axs[1, 1].set_title("Portfolio Budget")
for axs in axs.flat:
    axs.tick_params(axis="x", labelrotation = 90)

fig.tight_layout()


# In[68]:


# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html

#ungrouped value distribution
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(12,6))
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(clean["date"], clean["impressions"])
ax2.plot(clean["date"], clean["clicks"])
ax3.plot(clean["date"], clean["cost"])
ax4.plot(clean["date"], clean["portfolioBudget"])
ax4.set_ylim(0, 160)
for axs in (ax1,ax2,ax3,ax4):
    axs.tick_params(axis="x",labelrotation=90)


# # 13. separate dataframe subsets of representations
# ### Split into:
# #### 1. CTR 
# #### 2. CPC
# #### 3. introduce the lagged windows
# 
# ##### These dataframes will each be dealt with separately one at a time for the full process and will be numbered as chapters

# ##### 13.0 Encoding the outcome feature

# In[69]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
clean[["biddingStrategyType"]] = le.fit_transform(np.array(clean.biddingStrategyType)).astype("float64")


# ### Displaying the original set of features from which feature representations will be calculated

# In[70]:


clean.head() #before creating subsets


# ### Manually removing a campaign not found in the ground truth labels

# In[71]:


clean.drop(clean[clean.campaignId == 419744056].index, inplace=True)
clean.reset_index(drop=True,inplace=True)
clean


# ## 13.1 CTR dataframe
# #### Chapter 1A1 - calculating the CTR feature, dropping clicks and impressions features

# In[72]:


clean_CTR = clean.copy(deep=True)
clean_CTR["CTR"] = clean_CTR["clicks"]/clean_CTR["impressions"]
clean_CTR = clean_CTR.replace(np.nan, 0.00).round(2)
clean_CTR = clean_CTR.drop(["impressions","clicks"],axis=1)
clean_CTR.head()


# In[73]:


clean_CTR[clean_CTR["CTR"]==clean_CTR["CTR"].max()]


# #### Chapter 1A2 - solving an infinity problem

# In[74]:


# CTR infinity put to 2.0
# https://stackoverflow.com/a/62276558
clean_CTR["CTR"].replace(np.inf, 2.0, inplace=True)
clean_CTR[clean_CTR["CTR"]==clean_CTR["CTR"].max()]


# In[ ]:





# #### Chapter 2A - visualizations

# In[78]:


ccc = clean_CTR.copy(deep=True)
ccc = ccc.sort_values(by='date').groupby('campaignId').apply(lambda x: x[(x['campaignId'].values == 7941930)])
ccc.reset_index(drop=True)
ccc.groupby("date")['CTR'].mean().plot(kind='bar',figsize=(15,3),rot=45)


# In[79]:


fig,ax = plt.subplots(figsize=(15,3))
for campaignId in clean_CTR['campaignId'][:2]:
    ccc = clean_CTR[clean_CTR.campaignId==campaignId]
    ccc.groupby("date")['CTR'].mean().plot.hist(bins=10,figsize=(15,3),rot=45,title="Daily CTR")


# In[80]:


fig,ax = plt.subplots(figsize=(15,3))
for campaignId in clean_CTR['campaignId'][:5]:
    ccc = clean_CTR[clean_CTR.campaignId==campaignId]
    sns.kdeplot(ccc.groupby("date")['CTR'].mean(), shade=True)


# In[82]:


# https://www.kaggle.com/python10pm/plotting-with-python-learn-80-plots-step-by-step
fig = plt.figure(figsize = (15,5))
for campaignId in clean_CTR['campaignId'][7:8]:
    sns.distplot(ccc.groupby("date")['CTR'].mean(), kde = True, label = "CTR")


# In[83]:


fig,ax = plt.subplots(figsize=(14,6))
for campaignId in clean_CTR['campaignId'][4:6]:
    ccc = clean_CTR[clean_CTR.campaignId==campaignId]
    sns.distplot(ccc.groupby("date")['CTR'].mean())


# In[84]:


ccc = clean_CTR.copy(deep=True)
ccc = ccc.sort_values(by='date').groupby('campaignId').apply(lambda x: x[(x['campaignId'].values == 7941930)])
ccc.reset_index(drop=True)
ccc.groupby("date")['CTR'].mean().plot.hist(bins=10,figsize=(15,3),rot=45)


# In[85]:


ccc = clean_CTR.copy(deep=True)
ccc = ccc.sort_values(by='date').groupby('campaignId').apply(lambda x: x[(x['campaignId'].values == 419755576)])
ccc.reset_index(drop=True)
ccc.groupby("date")['CTR'].mean().plot(kind='bar',figsize=(15,3),rot=45)


# In[86]:


ccc = clean_CTR.copy(deep=True)
ccc = ccc.sort_values(by='date').groupby('campaignId').apply(lambda x: x[(x['campaignId'].values == 419755576)])
ccc.reset_index(drop=True)
ccc.groupby("date")['CTR'].mean().plot.hist(bins=10,figsize=(15,3),rot=45)


# In[87]:


ccc = clean_CTR.copy(deep=True)
ccc = ccc.sort_values(by='date').groupby('campaignId').apply(lambda x: x[(x['campaignId'].values == 420267976)])
ccc.reset_index(drop=True)
ccc.groupby("date")['CTR'].mean().plot(kind='bar',figsize=(15,3),rot=45)


# In[88]:


ccc = clean_CTR.copy(deep=True)
ccc = ccc.sort_values(by='date').groupby('campaignId').apply(lambda x: x[(x['campaignId'].values == 420267976)])
ccc.reset_index(drop=True)
ccc.groupby("date")['CTR'].mean().plot.hist(bins=10,figsize=(15,3),rot=45)


# In[89]:


ccc = clean_CTR.copy(deep=True)
ccc = ccc.sort_values(by='date').groupby('campaignId').apply(lambda x: x[(x['campaignId'].values == 684391125)])
ccc.reset_index(drop=True)
ccc.groupby("date")['CTR'].mean().plot(kind='bar',figsize=(15,3),rot=45)


# In[90]:


ccc = clean_CTR.copy(deep=True)
ccc = ccc.sort_values(by='date').groupby('campaignId').apply(lambda x: x[(x['campaignId'].values == 684391125)])
ccc.reset_index(drop=True)
ccc.groupby("date")['CTR'].mean().plot.hist(bins=10,figsize=(15,3),rot=45)


# In[91]:


import seaborn as sns; sns.set()
sns.set_style("ticks")

sns.pairplot(plots, hue="biddingStrategyType")


# In[ ]:





# #### df correlation

# In[92]:


from scipy.stats import spearmanr
# calculate Spearman's correlation
corr, a = spearmanr(clean_CTR["biddingStrategyType"], clean_CTR["cost"])
corr2, a = spearmanr(clean_CTR["CTR"], clean_CTR["cost"])

corr
print('Spearmans correlation: %.3f' % corr)
print('Spearmans correlation2: %.3f' % corr2)
clean_CTR.corr()


# In[ ]:





# ### Chapter 3 creating dataframes

# In[ ]:


#names: base, clean_CTR, CTR_lag1, CTR_lag7, CPC, CPC_lag1, CPC_lag7


# ### Base features separation into a new dataframe 

# In[93]:


base = clean.copy(deep=True)
base.head(2)


# In[94]:


#portfolioBudget does not change through time as it is set and left for consistency in the Google AI 
fig,ax = plt.subplots(figsize=(15,3))
for campaignId in base['campaignId'][:2]:
    t = base[base.campaignId==campaignId]
    t.groupby("date")['portfolioBudget'].mean().plot.hist(bins=10, figsize=(7,3),rot=45,title="Individual Campaign Budget")


# In[ ]:





# ### CPC dataframe and lags

# In[95]:


#### calculating the CPC feature
CPC = clean.copy(deep=True)
CPC['CPC'] = round(CPC['cost']/CPC['clicks'],2)
CPC.drop(['clicks','cost'],axis=1,inplace=True)


# In[96]:


# CPC infinity put to 2.0
# https://stackoverflow.com/a/62276558
CPC['CPC'].replace(np.inf, 2.0, inplace=True)
CPC[CPC['CPC']==CPC['CPC'].max()]


# In[97]:


fig,ax = plt.subplots(figsize=(15,3))
for campaignId in CPC['campaignId'][:5]:
    c = CPC[CPC.campaignId==campaignId]
    sns.kdeplot(c.groupby("date")['CPC'].mean(), shade=True)


# In[ ]:





# ### CPC_lag1

# In[98]:


#### calculating the CPC lag 1 feature

CPC_lag1 = CPC.copy(deep=True)

CPC_lag1['CPC_lag1'] = CPC_lag1.groupby(['campaignId'])['CPC'].shift(1)
CPC_lag1['CPC_lag1'] = CPC_lag1['CPC_lag1'].fillna(0)
CPC_lag1 = CPC_lag1.drop(['CPC'], axis=1)
CPC_lag1.head(2)


# In[99]:


fig,ax = plt.subplots(figsize=(15,3))
for campaignId in CPC_lag1['campaignId'][:5]:
    c = CPC_lag1[CPC_lag1.campaignId==campaignId]
    sns.kdeplot(c.groupby("date")['CPC_lag1'].mean(), shade=True)


# In[ ]:





# ### CPC_lag7

# In[100]:


#### calculating the CPC lag 7 feature
CPC_lag7 = CPC.copy(deep=True)

CPC_lag7['CPC_lag7'] = CPC_lag7.groupby(['campaignId'])['CPC'].shift(7)
CPC_lag7['CPC_lag7'] = CPC_lag7['CPC_lag7'].fillna(0)
CPC_lag7 = CPC_lag7.drop(['CPC'], axis=1)
CPC_lag7.head(2)


# In[101]:


fig,ax = plt.subplots(figsize=(15,3))
for campaignId in CPC_lag7['campaignId'][:5]:
    c = CPC_lag7[CPC_lag7.campaignId==campaignId]
    sns.kdeplot(c.groupby("date")['CPC_lag7'].mean(), shade=True)


# In[ ]:





# In[ ]:





# #### Chapter 3A2 - replace NaNs with zeros

# In[102]:


clean_CTR[["cost","portfolioBudget","CTR"]] = clean_CTR[["cost","portfolioBudget","CTR"]].replace(np.nan, 0.0).round(2)
clean_CTR.head(2)


# #### Chapter 3A3 - plot 

# In[103]:


#ungrouped
fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(14,4))
#fig.suptitle('Aligning x-axis using sharex')
ax1.plot(clean_CTR[["date"]],clean_CTR[["cost"]])
ax2.plot(clean_CTR[["date"]],clean_CTR[["CTR"]])
ax3.plot(clean_CTR[["date"]],clean_CTR[["portfolioBudget"]])
ax3.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)


# In[104]:


fig,ax = plt.subplots(figsize=(15,3))
for campaignId in clean_CTR['campaignId'][:5]:
    c = clean_CTR[clean_CTR.campaignId==campaignId]
    sns.kdeplot(c.groupby("date")['CTR'].mean(), shade=True)


# ### Chapter 4A - splitting, training the data, and storing predictions

# In[105]:


clean_CTR.info()


# In[ ]:





# ### 4A1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[107]:


# https://datascience.stackexchange.com/a/78426
import math
from sklearn.ensemble import IsolationForest
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

minmax = MinMaxScaler(feature_range=(0, 1))

# Create a dataframe to contain columns with anomalies
CTR_preds = clean_CTR.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = clean_CTR["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(clean_CTR['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(clean_CTR.values):
        df_train, df_test = clean_CTR.iloc[train_index], clean_CTR.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Isolation Forest - new indices train and evaluation")
        isof = IsolationForest(random_state=123, max_samples='auto', contamination=0.01, 
                       max_features=1.0, bootstrap=False, n_jobs=-1, verbose=0).fit(df_train_final)
        y_pred_isof = isof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTR_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTR_preds[column_name][index] = y_pred_isof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 4A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[839]:


CTR_preds["repeat2_fold2_timesplit3"].unique()


# ### 4A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[108]:


def anomaly_result(row):     
    row2 = row.drop(["date", "campaignId", "biddingStrategyType", "cost", "portfolioBudget", "CTR"])          
    result = 0
    for (name, data) in row2.iteritems():     
        timesplit = int(name[23:])         
        if data is not None:             
            column_value = timesplit * 0.2 * data         
        else:             
            column_value = 0
        result += column_value
    return round(result,2)


# In[109]:


CTR_preds["anomaly"] = CTR_preds.apply(lambda x: anomaly_result(x), axis=1)
CTR_preds.head()


# In[110]:


CTR_preds.anomaly.unique()


# ### 4A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[111]:


# which exact campaigns have anomaly score lower than or equal to -2 
CTR_preds[CTR_preds["anomaly"]<=-2]["campaignId"].value_counts()


# In[112]:


# which exact campaigns have anomaly score lower than or equal to -2 
CTR_preds[CTR_preds["anomaly"]<=-2]["campaignId"].unique()


# In[113]:


CTR_preds.head(2)


# In[114]:


CTR_isof = CTR_preds.copy(deep=True)


# In[115]:


def anomaly_percent(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CTR","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data == -1:
                negatives +=1
            elif data == 1:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    
    return negatives/(positives+negatives)


# In[116]:


CTR_isof["anom_percent"] = CTR_isof.apply(lambda x: anomaly_percent(x), axis=1)


# ### 4A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% score.

# In[117]:


CTR_isof["anom_percent"].unique()


# Number of unique campaigns which have a 100% anomaly score across evaluated splits.

# In[118]:


CTR_isof[CTR_isof["anom_percent"]==1.]["campaignId"].value_counts()


# ## All model and dataframe detection scores per date and campaign will be stored in this df

# In[119]:


CTR_isof = CTR_isof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
CTR_isof.columns = ['date','campaignId','anomaly_CTR_isof','anom_percent_CTR_isof']
CTR_isof.head(2)


# In[ ]:





# In[ ]:





# In[ ]:


######


# ### CTR LOF

# ### 4B1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. LOF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[120]:


import math
from sklearn.neighbors import LocalOutlierFactor
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTR_LOF_preds = clean_CTR.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = clean_CTR["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(clean_CTR['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(clean_CTR.values):
        df_train, df_test = clean_CTR.iloc[train_index], clean_CTR.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Local Outlier Factor - new indices train and evaluation")
        lof = LocalOutlierFactor(n_neighbors=20,novelty=False).fit_predict(df_train_final,df_test_final)
        #y_pred_lof = lof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTR_LOF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTR_LOF_preds[column_name][index] = lof[own_index]#y_pred_lof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 4B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[121]:


CTR_LOF_preds[CTR_LOF_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 4B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[122]:


CTR_LOF_preds["anomaly"] = CTR_LOF_preds.apply(lambda x: anomaly_result(x), axis=1)
CTR_LOF_preds.head()


# In[123]:


# used as a table in the thesis
CTR_LOF_preds.tail(20).to_excel("CTRLOFtable.xlsx")


# ### 4B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[124]:


CTR_preds[CTR_preds["anomaly"]<=-2]["campaignId"].value_counts()


# ### 4B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# In[125]:


CTR_LOF = CTR_LOF_preds.copy(deep=True)
CTR_LOF["anom_percent"] = CTR_LOF.apply(lambda x: anomaly_percent(x), axis=1)


# Number of unique campaigns which have a 100% anomaly score across evaluated splits.

# In[126]:


print(len(CTR_LOF[CTR_LOF["anom_percent"]==1.]["campaignId"].unique()))
CTR_LOF[CTR_LOF["anom_percent"]==1.]["campaignId"].unique()


# ### merge dataframe columns on date and ID as with isof

# In[127]:


lofcols = CTR_LOF[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
lofcols.columns = ['date','campaignId','anomaly_CTR_lof','anom_percent_CTR_lof']
lofcols.head(2)


# In[ ]:





# ### CTR Elliptic Envelope

# ### 4C1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EE model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[128]:


import math
from sklearn.covariance import EllipticEnvelope
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTR_EE_preds = clean_CTR.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = clean_CTR["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(clean_CTR['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(clean_CTR.values):
        df_train, df_test = clean_CTR.iloc[train_index], clean_CTR.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Elliptic Envelope - new indices train and evaluation")
        ee = EllipticEnvelope(random_state=0,support_fraction=1.0).fit(df_train_final)
        y_pred_ee = ee.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTR_EE_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTR_EE_preds[column_name][index] = y_pred_ee[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 4C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[129]:


CTR_EE_preds[CTR_EE_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 4C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[130]:


CTR_EE_preds["anomaly"] = CTR_EE_preds.apply(lambda x: anomaly_result(x), axis=1)
CTR_EE_preds.head()


# ### 4C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[131]:


len(CTR_EE_preds[CTR_EE_preds["anomaly"]<=-2]["campaignId"].unique())


# In[132]:


CTR_EE_preds[CTR_EE_preds["anomaly"]<=-2]["campaignId"].value_counts()


# In[133]:


CTR_EE = CTR_EE_preds.copy(deep=True)
CTR_EE["anom_percent"] = CTR_EE.apply(lambda x: anomaly_percent(x), axis=1)


# Number of unique campaigns which have a 100% anomaly score across evaluated splits.

# In[134]:


print(len(CTR_EE[CTR_EE["anom_percent"]==1.]["campaignId"].unique()))
CTR_EE[CTR_EE["anom_percent"]==1.]["campaignId"].unique()


# ### 4C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# In[135]:


eecols = CTR_EE[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
eecols.columns = ['date','campaignId','anomaly_CTR_ee','anom_percent_CTR_ee']
eecols.head(2)


# In[511]:


#################


# In[ ]:





# In[ ]:





# ### CTR Extended Isolation Forest

# ### 4D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[136]:


import math
import eif as iso
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTR_EIF_preds = clean_CTR.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = clean_CTR["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(clean_CTR['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(clean_CTR.values):
        df_train, df_test = clean_CTR.iloc[train_index], clean_CTR.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Extended Isolation Forest - new indices train and evaluation")
        eif = iso.iForest(df_train_final.to_numpy(),ntrees=100, sample_size=512, ExtensionLevel=1)
        y_pred_eif = eif.compute_paths(df_test_final.to_numpy())
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTR_EIF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTR_EIF_preds[column_name][index] = y_pred_eif[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# Anomalies are those that occur less frequently, hence, the number of points with higher anomaly scores reduces as the score increases

# In[137]:


CTR_EIF_preds["repeat2_fold2_timesplit3"].value_counts()


# In[138]:


CTR_EIF_preds["repeat2_fold2_timesplit3"].value_counts()[CTR_EIF_preds["repeat2_fold2_timesplit3"].value_counts()>10]


# In[139]:


print(round(CTR_EIF_preds["repeat2_fold2_timesplit3"].min(),2))
print(round(CTR_EIF_preds["repeat2_fold2_timesplit3"].max(),2))
print(round(CTR_EIF_preds["repeat2_fold2_timesplit3"].mean(),2))
print(round(CTR_EIF_preds["repeat2_fold2_timesplit3"].median(),2))


# In[140]:


print(round(CTR_EIF_preds["repeat2_fold2_timesplit6"].min(),2))
print(round(CTR_EIF_preds["repeat2_fold2_timesplit6"].max(),2))
print(round(CTR_EIF_preds["repeat2_fold2_timesplit6"].mean(),2))
print(round(CTR_EIF_preds["repeat2_fold2_timesplit6"].median(),2))


# In[141]:


print(round(CTR_EIF_preds["repeat2_fold3_timesplit6"].min(),2))
print(round(CTR_EIF_preds["repeat2_fold3_timesplit6"].max(),2))
print(round(CTR_EIF_preds["repeat2_fold3_timesplit6"].mean(),2))
print(round(CTR_EIF_preds["repeat2_fold3_timesplit6"].median(),2))


# In[142]:


print(round(CTR_EIF_preds["repeat2_fold1_timesplit5"].min(),2))
print(round(CTR_EIF_preds["repeat2_fold1_timesplit5"].max(),2))
print(round(CTR_EIF_preds["repeat2_fold1_timesplit5"].mean(),2))
print(round(CTR_EIF_preds["repeat2_fold1_timesplit5"].median(),2))


# ### 4D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[143]:


CTR_EIF_preds[CTR_EIF_preds["repeat2_fold2_timesplit3"]<=1]


# ### 4D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[144]:


CTR_EIF_preds["anomaly"] = CTR_EIF_preds.apply(lambda x: anomaly_result(x), axis=1)
CTR_EIF_preds.head()


# In[145]:


print(CTR_EIF_preds["anomaly"].min())
print(CTR_EIF_preds["anomaly"].max())
print((CTR_EIF_preds["anomaly"]>=1.6).value_counts())
print((CTR_EIF_preds["anomaly"]<0.49).value_counts())
print(CTR_EIF_preds["anomaly"].mean())
print(CTR_EIF_preds["anomaly"].median())
print(CTR_EIF_preds["anomaly"].value_counts())


# ### 4D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.

# In[146]:


CTR_EIF_preds[CTR_EIF_preds["anomaly"]>=(CTR_EIF_preds["anomaly"].max()-0.2)]["campaignId"].value_counts()


# ### 4D5 column containing percentage of anomaly per column in split + column containing anomaly score = save into csv

# In[147]:


def anomaly_percent_EIF(row):    
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CTR","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data >= 1.6:  #higher scores in the continuous distribution indicate anomaly
                negatives +=1
            elif data < 1.6:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    
    return negatives/(positives+negatives)


# In[148]:


CTR_EIF = CTR_EIF_preds.copy(deep=True)
CTR_EIF["anom_percent"] = CTR_EIF.apply(lambda x: anomaly_percent_EIF(x), axis=1)


# In[149]:


CTR_EIF.head()


# In[ ]:





# ### merge dataframe columns on date and ID as with isof

# In[150]:


eifcols = CTR_EIF[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
eifcols.columns = ['date','campaignId','anomaly_CTR_eif','anom_percent_CTR_eif']
eifcols.head(3)


# In[ ]:





# In[ ]:





# ## CTR lag 1 dataframe
# #### Chapter 5A - show the dataframe

# In[152]:


CTR_lag1 = clean_CTR.copy(deep=True)
CTR_lag1.head()


# In[153]:


#### calculating the CTR lag 1 feature

CTR_lag1['CTR_lag1'] = CTR_lag1.groupby(['campaignId'])['CTR'].shift(1)
CTR_lag1['CTR_lag1'] = CTR_lag1['CTR_lag1'].fillna(0)
CTR_lag1 = CTR_lag1.drop(['CTR'], axis=1)
CTR_lag1.head(2)


# In[154]:


fig,ax = plt.subplots(figsize=(15,3))
for campaignId in CTR_lag1['campaignId'][:5]:
    c = CTR_lag1[CTR_lag1.campaignId==campaignId]
    sns.kdeplot(c.groupby("date")['CTR_lag1'].mean(), shade=True)


# In[ ]:





# ## CTR lag 1 model application
# ### Chapter 5A1 Isolation Forest on CTR lag 1
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
# 

# In[155]:


# https://datascience.stackexchange.com/a/78426
import math
from sklearn.ensemble import IsolationForest
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTRlag1_preds = CTR_lag1.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CTR_lag1["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CTR_lag1['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CTR_lag1.values):
        df_train, df_test = CTR_lag1.iloc[train_index], CTR_lag1.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Isolation Forest - new indices train and evaluation")
        isof = IsolationForest(random_state=123, max_samples='auto', contamination=0.01, 
                       max_features=1.0, bootstrap=False, n_jobs=-1, verbose=0).fit(df_train_final)
        y_pred_isof = isof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTRlag1_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTRlag1_preds[column_name][index] = y_pred_isof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 5A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[156]:


CTRlag1_preds[CTRlag1_preds["repeat2_fold2_timesplit3"]==-1]


# ### 5A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[157]:


def anomaly_result_LAG1(row):     
    row2 = row.drop(["date", "campaignId", "biddingStrategyType", "cost", "portfolioBudget", "CTR_lag1"])          
    result = 0
    for (name, data) in row2.iteritems():     
        timesplit = int(name[23:])         
        if data is not None:             
            column_value = timesplit * 0.2 * data         
        else:             
            column_value = 0
        result += column_value
    return round(result,2)


# In[158]:


CTRlag1_preds["anomaly"] = CTRlag1_preds.apply(lambda x: anomaly_result_LAG1(x), axis=1)
CTRlag1_preds.head()


# ### 5A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[159]:


# which exact campaigns have anomaly score lower than or equal to -2 
CTRlag1_preds[CTRlag1_preds["anomaly"]<=-2.]["campaignId"].value_counts()


# ### 5A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# In[160]:


CTRlag1_isof = CTRlag1_preds.copy(deep=True)


# In[161]:


def anomaly_percent_LAG1(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CTR_lag1","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data == -1:
                negatives +=1
            elif data == 1:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    
    return negatives/(positives+negatives)


# In[162]:


CTRlag1_isof["anom_percent"] = CTRlag1_isof.apply(lambda x: anomaly_percent_LAG1(x), axis=1)
CTRlag1_isof.head()


# Value of 1 indicates 100% anomaly score.

# In[163]:


CTRlag1_isof[CTRlag1_isof["anom_percent"]==1]['campaignId'].unique()


# ### 5A6 save to csv based on date and ID, only anomaly and anom score

# In[164]:


CTRlag1isof = CTRlag1_isof[['date','campaignId','anomaly','anom_percent']]
CTRlag1isof.columns = ['date','campaignId','anomaly_CTRlag1_isof','anom_percent_CTRlag1_isof']
CTRlag1isof.head(2)


# In[110]:


#####


# ### Chapter 5B1 Local Outlier Factor on CTR lag 1
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[165]:


import math
from sklearn.neighbors import LocalOutlierFactor
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTRlag1LOF_preds = CTR_lag1.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CTR_lag1["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CTR_lag1['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CTR_lag1.values):
        df_train, df_test = CTR_lag1.iloc[train_index], CTR_lag1.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Local Outlier Factor - new indices train and evaluation")
        lof = LocalOutlierFactor(n_neighbors=20,novelty=False).fit_predict(df_train_final,df_test_final)
        #y_pred_lof = lof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTRlag1LOF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTRlag1LOF_preds[column_name][index] = lof[own_index]#y_pred_lof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 5B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[166]:


CTRlag1LOF_preds[CTRlag1LOF_preds["repeat2_fold2_timesplit3"]==-1]


# ### 5B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[167]:


CTRlag1LOF_preds["anomaly"] = CTRlag1LOF_preds.apply(lambda x: anomaly_result_LAG1(x), axis=1)
CTRlag1LOF_preds.head()


# ### 5B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[168]:


# which exact campaigns have anomaly score lower than or equal to -2 
CTRlag1LOF_preds[CTRlag1LOF_preds["anomaly"]<=-2.]["campaignId"].value_counts()


# In[169]:


CTRlag1_lof = CTRlag1LOF_preds.copy(deep=True)


# In[170]:


CTRlag1_lof["anom_percent"] = CTRlag1_lof.apply(lambda x: anomaly_percent_LAG1(x), axis=1)
CTRlag1_lof.head()


# ### 5B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[171]:


CTRlag1_lof[CTRlag1_lof["anom_percent"]==1]['campaignId'].unique()


# ### 5B6 save to csv based on date and ID, only anomaly and anom score

# In[172]:


CTRlag1lof = CTRlag1_lof[['date','campaignId','anomaly','anom_percent']]
CTRlag1lof.columns = ['date','campaignId','anomaly_CTRlag1_lof','anom_percent_CTRlag1_lof']
CTRlag1lof.head(2)


# In[907]:


#####


# In[ ]:





# ### Chapter 5C1 Elliptic Ellipse on CTR lag 1
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[173]:


import math
from sklearn.covariance import EllipticEnvelope
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTRlag1EE_preds = CTR_lag1.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CTR_lag1["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CTR_lag1['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CTR_lag1.values):
        df_train, df_test = CTR_lag1.iloc[train_index], CTR_lag1.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Elliptic Envelope - new indices train and evaluation")
        ee = EllipticEnvelope(random_state=0,support_fraction=1.0).fit(df_train_final)
        y_pred_ee = ee.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTRlag1EE_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTRlag1EE_preds[column_name][index] = y_pred_ee[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 5C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[174]:


CTRlag1EE_preds[CTRlag1EE_preds["repeat2_fold2_timesplit3"]==-1]


# ### 5C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[175]:


CTRlag1EE_preds["anomaly"] = CTRlag1EE_preds.apply(lambda x: anomaly_result_LAG1(x), axis=1)
CTRlag1EE_preds.head()


# ### 5C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[176]:


# which exact campaigns have anomaly score lower than or equal to -2 
len(CTRlag1EE_preds[CTRlag1EE_preds["anomaly"]<=-2.]["campaignId"].value_counts())


# In[177]:


CTRlag1_ee = CTRlag1EE_preds.copy(deep=True)
CTRlag1_ee["anom_percent"] = CTRlag1_ee.apply(lambda x: anomaly_percent_LAG1(x), axis=1)
CTRlag1_ee.head()


# ### 5C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[178]:


CTRlag1_ee[CTRlag1_ee["anom_percent"]==1]['campaignId'].unique()


# ### 5C6 save to csv based on date and ID, only anomaly and anom score

# In[179]:


CTRlag1eecols = CTRlag1_ee[['date','campaignId','anomaly','anom_percent']]
CTRlag1eecols.columns = ['date','campaignId','anomaly_CTRlag1_ee','anom_percent_CTRlag1_ee']
CTRlag1eecols.head(2)


# In[126]:


####


# ### CTR Lag 1 Extended Isolation Forest

# ### 5D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[180]:


import math
import eif as iso
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTRlag1EIF_preds = CTR_lag1.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CTR_lag1["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CTR_lag1['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CTR_lag1.values):
        df_train, df_test = CTR_lag1.iloc[train_index], CTR_lag1.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Extended Isolation Forest - new indices train and evaluation")
        eif = iso.iForest(df_train_final.to_numpy(),ntrees=100, sample_size=512, ExtensionLevel=1)
        y_pred_eif = eif.compute_paths(df_test_final.to_numpy())
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTRlag1EIF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTRlag1EIF_preds[column_name][index] = y_pred_eif[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# In[181]:


print(round(CTRlag1EIF_preds["repeat2_fold1_timesplit5"].max(),2))


# ### 5D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[182]:


CTRlag1EIF_preds[CTRlag1EIF_preds["repeat2_fold2_timesplit3"]<=1]


# ### 5D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[183]:


CTRlag1EIF_preds["anomaly"] = CTRlag1EIF_preds.apply(lambda x: anomaly_result_LAG1(x), axis=1)
CTRlag1EIF_preds.head()


# In[184]:


CTRlag1EIF_preds["anomaly"].max()


# ### 5D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.

# In[185]:


CTRlag1EIF_preds[CTRlag1EIF_preds["anomaly"]>=(CTRlag1EIF_preds["anomaly"].max()-0.2)]["campaignId"].value_counts()


# ### 5D5 column containing percentage per column in split + column containing anomaly score = save into csv

# In[186]:


# 1.81 is the max - 0.2
def anomaly_percent_EIF_LAG1(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CTR_lag1","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data >= 1.77:  #higher scores in the continuous distribution indicate anomaly
                negatives +=1
            elif data < 1.77:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    return negatives/(positives+negatives)


# In[187]:


CTRlag1_eif = CTRlag1EIF_preds.copy(deep=True)
CTRlag1_eif["anom_percent"] = CTRlag1_eif.apply(lambda x: anomaly_percent_EIF_LAG1(x), axis=1)


# In[188]:


CTRlag1_eif.head(3)


# ### 5D6 save to csv based on date and ID, only anomaly and anom score

# In[189]:


eifcolsCTR1 = CTRlag1_eif[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
eifcolsCTR1.columns = ['date','campaignId','anomaly_CTRlag1_eif','anom_percent_CTRlag1_eif']
eifcolsCTR1.head(3)


# In[ ]:


####


# In[ ]:





# In[ ]:





# ## CTR lag 7 dataframe
# #### Chapter 7A - show the dataframe

# In[191]:


CTR_lag7 = clean_CTR.copy(deep=True)
CTR_lag7.head()


# In[192]:


CTR_lag7['CTR_lag7'] = CTR_lag7.groupby(['campaignId'])['CTR'].shift(7)
CTR_lag7['CTR_lag7'] = CTR_lag7['CTR_lag7'].fillna(0)
CTR_lag7 = CTR_lag7.drop(['CTR'], axis=1)
CTR_lag7.head(2)


# In[193]:


fig,ax = plt.subplots(figsize=(15,3))
for campaignId in CTR_lag7['campaignId'][:5]:
    c = CTR_lag7[CTR_lag7.campaignId==campaignId]
    sns.kdeplot(c.groupby("date")['CTR_lag7'].mean(), shade=True)


# ## CTR lag 7 model application
# ### Chapter 5A1 Isolation Forest on CTR lag 7
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[194]:


# https://datascience.stackexchange.com/a/78426
import math
from sklearn.ensemble import IsolationForest
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTRlag7_preds = CTR_lag7.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CTR_lag7["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CTR_lag7['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CTR_lag7.values):
        df_train, df_test = CTR_lag7.iloc[train_index], CTR_lag7.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Isolation Forest - new indices train and evaluation")
        isof = IsolationForest(random_state=123, max_samples='auto', contamination=0.01, 
                       max_features=1.0, bootstrap=False, n_jobs=-1, verbose=0).fit(df_train_final)
        y_pred_isof = isof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTRlag7_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTRlag7_preds[column_name][index] = y_pred_isof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 6A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[195]:


CTRlag7_preds[CTRlag7_preds["repeat2_fold2_timesplit3"]==-1]


# ### 6A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[197]:


def anomaly_result_LAG7(row):     
    row2 = row.drop(["date", "campaignId", "biddingStrategyType", "cost", "portfolioBudget", "CTR_lag7"])          
    result = 0
    for (name, data) in row2.iteritems():     
        timesplit = int(name[23:])         
        if data is not None:             
            column_value = timesplit * 0.2 * data         
        else:             
            column_value = 0
        result += column_value
    return round(result,2)


# In[198]:


CTRlag7_preds["anomaly"] = CTRlag7_preds.apply(lambda x: anomaly_result_LAG7(x), axis=1)
CTRlag7_preds.head()


# ### 6A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[199]:


CTRlag7_preds[CTRlag7_preds["anomaly"]<=-2]["campaignId"].value_counts()


# ### 6A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# In[200]:


CTRlag7_isof = CTRlag7_preds.copy(deep=True)


# In[201]:


def anomaly_percent_LAG7(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CTR_lag7","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data == -1:
                negatives +=1
            elif data == 1:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    
    return negatives/(positives+negatives)


# In[202]:


CTRlag7_isof["anom_percent"] = CTRlag7_isof.apply(lambda x: anomaly_percent_LAG7(x), axis=1)
CTRlag7_isof.head()


# Value of 1 indicates 100% anomaly score.

# In[203]:


CTRlag7_isof[CTRlag7_isof["anom_percent"]==1]['campaignId'].unique()


# ### 6D6 save to csv based on date and ID, only anomaly and anom score

# In[204]:


isofcolsCTR7 = CTRlag7_isof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
isofcolsCTR7.columns = ['date','campaignId','anomaly_CTRlag7_isof','anom_percent_CTRlag7_isof']
isofcolsCTR7.head(3)


# In[938]:


####


# In[ ]:





# ### Chapter 6B1 Local Outlier Factor on CTR lag 7
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[205]:


import math
from sklearn.neighbors import LocalOutlierFactor
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTRlag7LOF_preds = CTR_lag7.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CTR_lag7["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CTR_lag7['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CTR_lag7.values):
        df_train, df_test = CTR_lag7.iloc[train_index], CTR_lag7.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Local Outlier Factor - new indices train and evaluation")
        lof = LocalOutlierFactor(n_neighbors=20,novelty=False).fit_predict(df_train_final,df_test_final)
        #y_pred_lof = lof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTRlag7LOF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTRlag7LOF_preds[column_name][index] = lof[own_index]#y_pred_lof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 6B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[206]:


CTRlag7LOF_preds[CTRlag7LOF_preds["repeat2_fold2_timesplit3"]==-1]


# ### 6B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[207]:


CTRlag7LOF_preds["anomaly"] = CTRlag7LOF_preds.apply(lambda x: anomaly_result_LAG7(x), axis=1)
CTRlag7LOF_preds.head()


# ### 6B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[208]:


# which exact campaigns have anomaly score lower than or equal to -2 
CTRlag7LOF_preds[CTRlag7LOF_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[209]:


CTRlag7_lof = CTRlag7LOF_preds.copy(deep=True)
CTRlag7_lof["anom_percent"] = CTRlag7_lof.apply(lambda x: anomaly_percent_LAG7(x), axis=1)
CTRlag7_lof.head()


# ### 6B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[210]:


CTRlag7_lof[CTRlag7_lof["anom_percent"]==1]['campaignId'].unique()


# ### 6B6 save to csv based on date and ID, only anomaly and anom score

# In[211]:


CTRlag7colslof = CTRlag7_lof[['date','campaignId','anomaly','anom_percent']]
CTRlag7colslof.columns = ['date','campaignId','anomaly_CTRlag7_lof','anom_percent_CTRlag7_lof']
CTRlag7colslof.head(2)


# In[ ]:


####


# In[ ]:





# In[ ]:





# ### Chapter 6C1 Elliptic Ellipse on CTR lag 7
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[212]:


import math
from sklearn.covariance import EllipticEnvelope
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTRlag7EE_preds = CTR_lag7.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CTR_lag7["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CTR_lag7['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CTR_lag7.values):
        df_train, df_test = CTR_lag7.iloc[train_index], CTR_lag7.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Elliptic Envelope - new indices train and evaluation")
        ee = EllipticEnvelope(random_state=0,support_fraction=1.0).fit(df_train_final)
        y_pred_ee = ee.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTRlag7EE_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTRlag7EE_preds[column_name][index] = y_pred_ee[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 6C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[213]:


CTRlag7EE_preds[CTRlag7EE_preds["repeat2_fold2_timesplit3"]==-1]


# ### 6C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[214]:


CTRlag7EE_preds["anomaly"] = CTRlag7EE_preds.apply(lambda x: anomaly_result_LAG7(x), axis=1)
CTRlag7EE_preds.head()


# ### 6C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[215]:


# which exact campaigns have anomaly score lower than or equal to -2 
CTRlag7EE_preds[CTRlag7EE_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[216]:


CTRlag7_ee = CTRlag7EE_preds.copy(deep=True)
CTRlag7_ee["anom_percent"] = CTRlag7_ee.apply(lambda x: anomaly_percent_LAG7(x), axis=1)
CTRlag7_ee.head()


# ### 6C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[217]:


CTRlag7_ee[CTRlag7_ee["anom_percent"]==1]['campaignId'].unique()


# ### 6C6 save to csv based on date and ID, only anomaly and anom score

# In[218]:


CTRlag7colsee = CTRlag7_ee[['date','campaignId','anomaly','anom_percent']]
CTRlag7colsee.columns = ['date','campaignId','anomaly_CTRlag7_ee','anom_percent_CTRlag7_ee']
CTRlag7colsee.head(2)


# In[ ]:


####


# In[ ]:





# In[ ]:





# ### CTR Lag 1 Extended Isolation Forest
# 
# ### 6D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[219]:


import math
import eif as iso
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CTRlag7EIF_preds = CTR_lag7.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CTR_lag7["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CTR_lag7['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CTR_lag7.values):
        df_train, df_test = CTR_lag7.iloc[train_index], CTR_lag7.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Extended Isolation Forest - new indices train and evaluation")
        eif = iso.iForest(df_train_final.to_numpy(),ntrees=100, sample_size=512, ExtensionLevel=1)
        y_pred_eif = eif.compute_paths(df_test_final.to_numpy())
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CTRlag7EIF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CTRlag7EIF_preds[column_name][index] = y_pred_eif[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# In[220]:


print(round(CTRlag7EIF_preds["repeat2_fold1_timesplit5"].max(),2))


# ### 6D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[221]:


CTRlag7EIF_preds[CTRlag7EIF_preds["repeat2_fold2_timesplit3"]<=1]


# ### 6D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[222]:


CTRlag7EIF_preds["anomaly"] = CTRlag7EIF_preds.apply(lambda x: anomaly_result_LAG7(x), axis=1)
CTRlag7EIF_preds.head()


# In[223]:


CTRlag7EIF_preds["anomaly"].max()


# ### 6D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.

# In[224]:


CTRlag7EIF_preds[CTRlag7EIF_preds["anomaly"]>=(CTRlag7EIF_preds["anomaly"].max()-0.2)]["campaignId"].unique()


# ### 6D5 column containing percentage of lower than (max-0.2) per column in split + column containing anomaly score = save into csv

# In[225]:


# 1.81 is the max - 0.2
def anomaly_percent_EIF_LAG7(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CTR_lag7","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data >= 1.61:  #higher scores in the continuous distribution indicate anomaly
                negatives +=1
            elif data < 1.61:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    return negatives/(positives+negatives)


# In[226]:


CTRlag7_eif = CTRlag7EIF_preds.copy(deep=True)
CTRlag7_eif["anom_percent"] = CTRlag7_eif.apply(lambda x: anomaly_percent_EIF_LAG7(x), axis=1)
CTRlag7_eif.head(3)


# ### 6D6 save to csv based on date and ID, only anomaly and anom score

# In[227]:


eifcolsCTR7 = CTRlag7_eif[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
eifcolsCTR7.columns = ['date','campaignId','anomaly_CTRlag7_eif','anom_percent_CTRlag7_eif']
eifcolsCTR7.head(3)


# In[ ]:


####


# In[ ]:





# In[ ]:





# ## CPC dataframe
# #### Chapter 7A - show the CPC dataframe

# In[228]:


CPC = clean.copy(deep=True)
CPC['CPC'] = round(CPC['clicks']/CPC['impressions'],2)
CPC.drop(['impressions','clicks'],axis=1,inplace=True)
CPC.head(3)


# In[229]:


CPC['CPC'].max()


# #### Solving an infinity problem

# In[230]:


# CPC infinity put to 2.0
# https://stackoverflow.com/a/62276558
CPC['CPC'].replace(np.inf, 2.0, inplace=True)
CPC[CPC['CPC']==CPC['CPC'].max()]


# #### Removing NaNs

# In[231]:


CPC['CPC'] = CPC['CPC'].fillna(0)
CPC.isna(), CPC.head()


# ## CPC model application
# ### Chapter 7A1 Isolation Forest on CPC
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[232]:


# https://datascience.stackexchange.com/a/78426
import math
from sklearn.ensemble import IsolationForest
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPC_preds = CPC.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC.values):
        df_train, df_test = CPC.iloc[train_index], CPC.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Isolation Forest - new indices train and evaluation")
        isof = IsolationForest(random_state=123, max_samples='auto', contamination=0.01, 
                       max_features=1.0, bootstrap=False, n_jobs=-1, verbose=0).fit(df_train_final)
        y_pred_isof = isof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPC_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPC_preds[column_name][index] = y_pred_isof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 7A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[233]:


CPC_preds[CPC_preds["repeat2_fold2_timesplit3"]==-1]


# ### 7A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[234]:


def anomaly_result_CPC(row):     
    row2 = row.drop(["date", "campaignId", "biddingStrategyType", "cost", "portfolioBudget", "CPC"])          
    result = 0
    for (name, data) in row2.iteritems():     
        timesplit = int(name[23:])         
        if data is not None:             
            column_value = timesplit * 0.2 * data         
        else:             
            column_value = 0
        result += column_value
    return round(result,2)


# In[235]:


CPC_preds["anomaly"] = CPC_preds.apply(lambda x: anomaly_result_CPC(x), axis=1)
CPC_preds.head()


# ### 7A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[236]:


CPC_preds[CPC_preds["anomaly"]<=-2]["campaignId"].unique()


# ### 7A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# In[237]:


CPC_isof = CPC_preds.copy(deep=True)


# In[238]:


def anomaly_percent_CPC(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CPC","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data == -1:
                negatives +=1
            elif data == 1:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    
    return negatives/(positives+negatives)


# In[239]:


CPC_isof["anom_percent"] = CPC_isof.apply(lambda x: anomaly_percent_CPC(x), axis=1)
CPC_isof.head()


# Value of 1 indicates 100% anomaly score.

# In[240]:


CPC_isof[CPC_isof["anom_percent"]==1]['campaignId'].unique()


# ### 7A6 save to csv based on date and ID, only anomaly and anom score

# In[241]:


CPCcols = CPC_isof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
CPCcols.columns =  ['date','campaignId','anomaly_CPC_isof','anom_percent_CPC_isof']
CPCcols.head(3)


# In[186]:


####


# In[ ]:





# In[ ]:





# ### Chapter 7B1 Local Outlier Factor on CPC
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[242]:


import math
from sklearn.neighbors import LocalOutlierFactor
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPCLOF_preds = CPC.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC.values):
        df_train, df_test = CPC.iloc[train_index], CPC.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Local Outlier Factor - new indices train and evaluation")
        lof = LocalOutlierFactor(n_neighbors=20,novelty=False).fit_predict(df_train_final,df_test_final)
        #y_pred_lof = lof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPCLOF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPCLOF_preds[column_name][index] = lof[own_index]#y_pred_lof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 7B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[243]:


CPCLOF_preds[CPCLOF_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 7B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[244]:


CPCLOF_preds["anomaly"] = CPCLOF_preds.apply(lambda x: anomaly_result_CPC(x), axis=1)
CPCLOF_preds.head()


# ### 7B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[245]:


# which exact campaigns have anomaly score lower than or equal to -2 
CPCLOF_preds[CPCLOF_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[246]:


CPC_lof = CPCLOF_preds.copy(deep=True)
CPC_lof["anom_percent"] = CPC_lof.apply(lambda x: anomaly_percent_CPC(x), axis=1)
CPC_lof.head()


# ### 7B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[247]:


CPC_lof[CPC_lof["anom_percent"]==1]['campaignId'].unique()


# ### 7B6 save to csv based on date and ID, only anomaly and anom score

# In[248]:


CPCcolslof = CPC_lof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
CPCcolslof.columns = ['date','campaignId','anomaly_CPC_lof','anom_percent_CPC_lof']
CPCcolslof.head(2)


# In[ ]:


####


# ### Chapter 7C1 Elliptic Ellipse on CPC
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[249]:


import math
from sklearn.covariance import EllipticEnvelope
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPCEE_preds = CPC.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC.values):
        df_train, df_test = CPC.iloc[train_index], CPC.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Elliptic Envelope - new indices train and evaluation")
        ee = EllipticEnvelope(random_state=0,support_fraction=1.0).fit(df_train_final)
        y_pred_ee = ee.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPCEE_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPCEE_preds[column_name][index] = y_pred_ee[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 7C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[250]:


CPCEE_preds[CPCEE_preds["repeat2_fold2_timesplit3"]==-1]


# ### 7C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[251]:


CPCEE_preds["anomaly"] = CPCEE_preds.apply(lambda x: anomaly_result_CPC(x), axis=1)
CPCEE_preds.head()


# ### 7C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[252]:


# which exact campaigns have anomaly score lower than or equal to -2 
CPCEE_preds[CPCEE_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[253]:


CPC_ee = CPCEE_preds.copy(deep=True)
CPC_ee["anom_percent"] = CPC_ee.apply(lambda x: anomaly_percent_CPC(x), axis=1)
CPC_ee.head()


# ### 7C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[254]:


CPC_ee[CPC_ee["anom_percent"]==1]['campaignId'].unique()


# ### 7C6 save to csv based on date and ID, only anomaly and anom score

# In[255]:


CPCcolsee = CPC_ee[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
CPCcolsee.columns = ['date','campaignId','anomaly_CPC_ee','anom_percent_CPC_ee']
CPCcolsee.head(2)


# In[ ]:


####


# ### CPC Extended Isolation Forest
# 
# ### 7D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[256]:


import math
import eif as iso
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPCEIF_preds = CPC.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC.values):
        df_train, df_test = CPC.iloc[train_index], CPC.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Extended Isolation Forest - new indices train and evaluation")
        eif = iso.iForest(df_train_final.to_numpy(),ntrees=100, sample_size=512, ExtensionLevel=1)
        y_pred_eif = eif.compute_paths(df_test_final.to_numpy())
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPCEIF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPCEIF_preds[column_name][index] = y_pred_eif[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# In[257]:


print(round(CPCEIF_preds["repeat2_fold1_timesplit5"].max(),2))


# ### 7D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[258]:


CPCEIF_preds[CPCEIF_preds["repeat2_fold2_timesplit3"]<=1]


# ### 7D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[259]:


CPCEIF_preds["anomaly"] = CPCEIF_preds.apply(lambda x: anomaly_result_CPC(x), axis=1)
CPCEIF_preds.head()


# In[260]:


CPCEIF_preds["anomaly"].max()


# ### 7D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.

# In[261]:


CPCEIF_preds[CPCEIF_preds["anomaly"]>=(CPCEIF_preds["anomaly"].max()-0.2)]["campaignId"].unique()


# ### 7D5 column containing percentage of less or equal to threshold per column in split + column containing anomaly score = save into csv

# In[262]:


# 1.8 is the max - 0.2
def anomaly_percent_EIF_CPC(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CPC","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data >= 1.8:  #higher scores in the continuous distribution indicate anomaly
                negatives +=1
            elif data < 1.8:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    return negatives/(positives+negatives)


# In[263]:


CPC_eif = CPCEIF_preds.copy(deep=True)
CPC_eif["anom_percent"] = CPC_eif.apply(lambda x: anomaly_percent_EIF_CPC(x), axis=1)
CPC_eif.head(3)


# ### 7D6 save to csv based on date and ID, only anomaly and anom score

# In[264]:


eifcolsCPC = CPC_eif[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
eifcolsCPC.columns = ['date','campaignId','anomaly_CPC_eif','anom_percent_CPC_eif']
eifcolsCPC.head(3)


# In[ ]:


####


# ## CPC lag 1 dataframe
# #### Chapter 8A - show the dataframe

# In[265]:


CPC_lag1 = CPC.copy(deep=True)
CPC_lag1.head(3)


# In[266]:


CPC_lag1['CPC_lag1'] = CPC_lag1.groupby(['campaignId'])['CPC'].shift(1)
CPC_lag1['CPC_lag1'] = CPC_lag1['CPC_lag1'].fillna(0)
CPC_lag1 = CPC_lag1.drop(['CPC'], axis=1)
CPC_lag1.head(2)


# ## CPC lag 1model application
# ### Chapter 8A1 Isolation Forest on CPC lag 1
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set.

# In[267]:


# https://datascience.stackexchange.com/a/78426
import math
from sklearn.ensemble import IsolationForest
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPClag1_preds = CPC_lag1.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC_lag1["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC_lag1['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC_lag1.values):
        df_train, df_test = CPC_lag1.iloc[train_index], CPC_lag1.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Isolation Forest - new indices train and evaluation")
        isof = IsolationForest(random_state=123, max_samples='auto', contamination=0.01, 
                       max_features=1.0, bootstrap=False, n_jobs=-1, verbose=0).fit(df_train_final)
        y_pred_isof = isof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPClag1_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPClag1_preds[column_name][index] = y_pred_isof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 8A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[268]:


CPClag1_preds[CPClag1_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 8A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[269]:


def anomaly_result_LAG1CPC(row):     
    row2 = row.drop(["date", "campaignId", "biddingStrategyType", "cost", "portfolioBudget", "CPC_lag1"])          
    result = 0
    for (name, data) in row2.iteritems():     
        timesplit = int(name[23:])         
        if data is not None:             
            column_value = timesplit * 0.2 * data         
        else:             
            column_value = 0
        result += column_value
    return round(result,2)


# In[270]:


CPClag1_preds["anomaly"] = CPClag1_preds.apply(lambda x: anomaly_result_LAG1CPC(x), axis=1)
CPClag1_preds.head()


# ### 8A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[271]:


CPClag1_preds[CPClag1_preds["anomaly"]<=-2]["campaignId"].unique()


# ### 8A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# In[272]:


CPClag1_isof = CPClag1_preds.copy(deep=True)


# In[273]:


def anomaly_percent_LAG1CPC(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CPC_lag1","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data == -1:
                negatives +=1
            elif data == 1:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    
    return negatives/(positives+negatives)


# In[274]:


CPClag1_isof["anom_percent"] = CPClag1_isof.apply(lambda x: anomaly_percent_LAG1CPC(x), axis=1)
CPClag1_isof.head()


# Value of 1 indicates 100% anomaly score.

# In[275]:


CPClag1_isof[CPClag1_isof["anom_percent"]==1]['campaignId'].unique()


# ### 8D6 save to csv based on date and ID, only anomaly and anom score

# In[276]:


isofcolsCPC1 = CPClag1_isof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
isofcolsCPC1.columns = ['date','campaignId','anomaly_CPClag1_isof','anom_percent_CPClag1_isof']
isofcolsCPC1.head(3)


# In[ ]:


####


# ### Chapter 8B1 Local Outlier Factor on CPC lag 1 
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[277]:


import math
from sklearn.neighbors import LocalOutlierFactor
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPClag1LOF_preds = CPC_lag1.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC_lag1["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC_lag1['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC_lag1.values):
        df_train, df_test = CPC_lag1.iloc[train_index], CPC_lag1.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Local Outlier Factor - new indices train and evaluation")
        lof = LocalOutlierFactor(n_neighbors=20,novelty=False).fit_predict(df_train_final,df_test_final)
        #y_pred_lof = lof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPClag1LOF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPClag1LOF_preds[column_name][index] = lof[own_index]#y_pred_lof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 8B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[278]:


CPClag1LOF_preds[CPClag1LOF_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 8B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[279]:


CPClag1LOF_preds["anomaly"] = CPClag1LOF_preds.apply(lambda x: anomaly_result_LAG1CPC(x), axis=1)
CPClag1LOF_preds.head()


# ### 8B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[280]:


# which exact campaigns have anomaly score lower than or equal to -2 
CPClag1LOF_preds[CPClag1LOF_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[281]:


CPClag1_lof = CPClag1LOF_preds.copy(deep=True)
CPClag1_lof["anom_percent"] = CPClag1_lof.apply(lambda x: anomaly_percent_LAG1CPC(x), axis=1)
CPClag1_lof.head()


# ### 8B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[282]:


CPClag1_lof[CPClag1_lof["anom_percent"]==1]['campaignId'].unique()


# In[283]:


### 8B6 save to csv based on date and ID, only anomaly and anom score
CPClag1colslof = CPClag1_lof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
CPClag1colslof.columns = ['date','campaignId','anomaly_CPClag1_lof','anom_percent_CPClag1_lof']
CPClag1colslof.head(2)


# In[ ]:


####


# In[ ]:





# In[ ]:





# ### Chapter 8C1 Elliptic Ellipse on CPC lag 1
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[284]:


import math
from sklearn.covariance import EllipticEnvelope
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPClag1EE_preds = CPC_lag1.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC_lag1["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC_lag1['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC_lag1.values):
        df_train, df_test = CPC_lag1.iloc[train_index], CPC_lag1.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Elliptic Envelope - new indices train and evaluation")
        ee = EllipticEnvelope(random_state=0,support_fraction=1.0).fit(df_train_final)
        y_pred_ee = ee.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPClag1EE_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPClag1EE_preds[column_name][index] = y_pred_ee[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 8C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[285]:


CPClag1EE_preds[CPClag1EE_preds["repeat2_fold2_timesplit3"]==-1]


# ### 8C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[286]:


CPClag1EE_preds["anomaly"] = CPClag1EE_preds.apply(lambda x: anomaly_result_LAG1CPC(x), axis=1)
CPClag1EE_preds.head()


# ### 8C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[287]:


# which exact campaigns have anomaly score lower than or equal to -2 
CPClag1EE_preds[CPClag1EE_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[288]:


CPClag1_ee = CPClag1EE_preds.copy(deep=True)
CPClag1_ee["anom_percent"] = CPClag1_ee.apply(lambda x: anomaly_percent_LAG1CPC(x), axis=1)
CPClag1_ee.head()


# ### 8C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[289]:


CPClag1_ee[CPClag1_ee["anom_percent"]==1]['campaignId'].unique()


# ### 8C6 save to csv based on date and ID, only anomaly and anom score

# In[290]:


CPClag1colsee = CPClag1_ee[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
CPClag1colsee.columns = ['date','campaignId','anomaly_CPClag1_ee','anom_percent_CPClag1_ee']
CPClag1colsee.head(2)


# In[236]:


####


# ### CPC Lag 1 Extended Isolation Forest
# 
# ### 8D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[291]:


import math
import eif as iso
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPClag1EIF_preds = CPC_lag1.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC_lag1["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC_lag1['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC_lag1.values):
        df_train, df_test = CPC_lag1.iloc[train_index], CPC_lag1.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Extended Isolation Forest - new indices train and evaluation")
        eif = iso.iForest(df_train_final.to_numpy(),ntrees=100, sample_size=512, ExtensionLevel=1)
        y_pred_eif = eif.compute_paths(df_test_final.to_numpy())
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPClag1EIF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPClag1EIF_preds[column_name][index] = y_pred_eif[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# In[292]:


print(round(CPClag1EIF_preds["repeat2_fold1_timesplit5"].max(),2))


# ### 8D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[293]:


CPClag1EIF_preds[CPClag1EIF_preds["repeat2_fold2_timesplit3"]<=1]


# ### 8D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[294]:


CPClag1EIF_preds["anomaly"] = CPClag1EIF_preds.apply(lambda x: anomaly_result_LAG1CPC(x), axis=1)
CPClag1EIF_preds.head()


# In[295]:


CPClag1EIF_preds["anomaly"].max()


# ### 8D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.

# In[296]:


CPClag1EIF_preds[CPClag1EIF_preds["anomaly"]>=(CPClag1EIF_preds["anomaly"].max()-0.2)]["campaignId"].unique()


# ### 8D5 column containing percentage of less or equal to threshold per column in split + column containing anomaly score = save into csv

# In[297]:


# 1.81 is the max - 0.2
def anomaly_percent_EIF_LAG1CPC(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CPC_lag1","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data >= 1.61:  #higher scores in the continuous distribution indicate anomaly
                negatives +=1
            elif data < 1.61:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    return negatives/(positives+negatives)


# In[298]:


CPClag1_eif = CPClag1EIF_preds.copy(deep=True)
CPClag1_eif["anom_percent"] = CPClag1_eif.apply(lambda x: anomaly_percent_EIF_LAG1CPC(x), axis=1)
CPClag1_eif.head(3)


# ### 8D6 save to csv based on date and ID, only anomaly and anom score

# In[299]:


eifcolsCPC1 = CPClag1_eif[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
eifcolsCPC1.columns = ['date','campaignId','anomaly_CPClag1_eif','anom_percent_CPClag1_eif']
eifcolsCPC1.head(3)


# In[ ]:


####


# ## CPC lag 7 dataframe
# #### Chapter 9A - show the dataframe

# In[300]:


CPC_lag7 = CPC.copy(deep=True)

CPC_lag7['CPC_lag7'] = CPC_lag7.groupby(['campaignId'])['CPC'].shift(7)
CPC_lag7['CPC_lag7'] = CPC_lag7['CPC_lag7'].fillna(0)
CPC_lag7 = CPC_lag7.drop(['CPC'], axis=1)
CPC_lag7.head(2)


# ## CPC lag 7 model application
# ### Chapter 9A1 Isolation Forest on CPC lag 7
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[301]:


# https://datascience.stackexchange.com/a/78426
import math
from sklearn.ensemble import IsolationForest
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPClag7_preds = CPC_lag7.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC_lag7["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC_lag7['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC_lag7.values):
        df_train, df_test = CPC_lag7.iloc[train_index], CPC_lag7.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Isolation Forest - new indices train and evaluation")
        isof = IsolationForest(random_state=123, max_samples='auto', contamination=0.01, 
                       max_features=1.0, bootstrap=False, n_jobs=-1, verbose=0).fit(df_train_final)
        y_pred_isof = isof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPClag7_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPClag7_preds[column_name][index] = y_pred_isof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 9A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[302]:


CPClag7_preds[CPClag7_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 9A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[303]:


def anomaly_result_LAG7CPC(row):     
    row2 = row.drop(["date", "campaignId", "biddingStrategyType", "cost", "portfolioBudget", "CPC_lag7"])          
    result = 0
    for (name, data) in row2.iteritems():     
        timesplit = int(name[23:])         
        if data is not None:             
            column_value = timesplit * 0.2 * data         
        else:             
            column_value = 0
        result += column_value
    return round(result,2)


# In[304]:


CPClag7_preds["anomaly"] = CPClag7_preds.apply(lambda x: anomaly_result_LAG7CPC(x), axis=1)
CPClag7_preds.head()


# ### 9A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[305]:


CPClag7_preds[CPClag7_preds["anomaly"]<=-2]["campaignId"].unique()


# ### 9A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# In[306]:


def anomaly_percent_LAG7CPC(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CPC_lag7","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data == -1:
                negatives +=1
            elif data == 1:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    
    return negatives/(positives+negatives)


# In[307]:


CPClag7_isof = CPClag7_preds.copy(deep=True)
CPClag7_isof["anom_percent"] = CPClag7_isof.apply(lambda x: anomaly_percent_LAG7CPC(x), axis=1)
CPClag7_isof.head()


# Value of 1 indicates 100% anomaly score.

# In[308]:


CPClag7_isof[CPClag7_isof["anom_percent"]==1]['campaignId'].unique()


# ### 9D6 save to csv based on date and ID, only anomaly and anom score

# In[309]:


isofcolsCPC7 = CPClag7_isof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
isofcolsCPC7.columns =  ['date','campaignId','anomaly_CPClag7_isof','anom_percent_CPClag7_isof']
isofcolsCPC7.head(3)


# In[ ]:


####


# ### Chapter 9B1 Local Outlier Factor on CPC lag 7
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[310]:


import math
from sklearn.neighbors import LocalOutlierFactor
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPClag7LOF_preds = CPC_lag7.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC_lag7["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC_lag7['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC_lag7.values):
        df_train, df_test = CPC_lag7.iloc[train_index], CPC_lag7.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Local Outlier Factor - new indices train and evaluation")
        lof = LocalOutlierFactor(n_neighbors=20,novelty=False).fit_predict(df_train_final,df_test_final)
        #y_pred_lof = lof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPClag7LOF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPClag7LOF_preds[column_name][index] = lof[own_index]#y_pred_lof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 9B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[311]:


CPClag7LOF_preds[CPClag7LOF_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 9B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[312]:


CPClag7LOF_preds["anomaly"] = CPClag7LOF_preds.apply(lambda x: anomaly_result_LAG7CPC(x), axis=1)
CPClag7LOF_preds.head()


# ### 9B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[313]:


# which exact campaigns have anomaly score lower than or equal to -2 
CPClag7LOF_preds[CPClag7LOF_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[314]:


CPClag7_lof = CPClag7_preds.copy(deep=True)
CPClag7_lof["anom_percent"] = CPClag7_lof.apply(lambda x: anomaly_percent_LAG7CPC(x), axis=1)
CPClag7_lof.head()


# ### 9B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[315]:


CPClag7_lof[CPClag7_lof["anom_percent"]==1]['campaignId'].unique()


# ### 9B6 save to csv based on date and ID, only anomaly and anom score

# In[316]:


CPClag7colslof = CPClag7_lof[['date','campaignId','anomaly','anom_percent']]
CPClag7colslof.columns = ['date','campaignId','anomaly_CPClag7_lof','anom_percent_CPClag7_lof']
CPClag7colslof.head(2)


# In[1051]:


####


# ### Chapter 9C1 Elliptic Ellipse on CPC lag 7
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[317]:


import math
from sklearn.covariance import EllipticEnvelope
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPClag7EE_preds = CPC_lag7.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC_lag7["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC_lag7['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC_lag7.values):
        df_train, df_test = CPC_lag7.iloc[train_index],CPC_lag7.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Elliptic Envelope - new indices train and evaluation")
        ee = EllipticEnvelope(random_state=0,support_fraction=1.0).fit(df_train_final)
        y_pred_ee = ee.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPClag7EE_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPClag7EE_preds[column_name][index] = y_pred_ee[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 9C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[318]:


CPClag7EE_preds[CPClag7EE_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 9C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[319]:


CPClag7EE_preds["anomaly"] = CPClag7EE_preds.apply(lambda x: anomaly_result_LAG7CPC(x), axis=1)
CPClag7EE_preds.head()


# ### 9C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[320]:


# which exact campaigns have anomaly score lower than or equal to -2 
CPClag7EE_preds[CPClag7EE_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[321]:


CPClag7_ee = CPClag7EE_preds.copy(deep=True)
CPClag7_ee["anom_percent"] = CPClag7_ee.apply(lambda x: anomaly_percent_LAG7CPC(x), axis=1)
CPClag7_ee.head()


# ### 9C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[322]:


CPClag7_ee[CPClag7_ee["anom_percent"]==1]['campaignId'].unique()


# ### 9C6 save to csv based on date and ID, only anomaly and anom score

# In[323]:


CPClag7colsee = CPClag7_ee[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
CPClag7colsee.columns = ['date','campaignId','anomaly_CPClag7_ee','anom_percent_CPClag7_ee']
CPClag7colsee.head(2)


# In[ ]:


####


# ### CPC Lag 7 Extended Isolation Forest
# 
# ### 9D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[324]:


import math
import eif as iso
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
CPClag7EIF_preds = CPC_lag7.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = CPC_lag7["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(CPC_lag7['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(CPC_lag7.values):
        df_train, df_test = CPC_lag7.iloc[train_index], CPC_lag7.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Extended Isolation Forest - new indices train and evaluation")
        eif = iso.iForest(df_train_final.to_numpy(),ntrees=100, sample_size=512, ExtensionLevel=1)
        y_pred_eif = eif.compute_paths(df_test_final.to_numpy())
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        CPClag7EIF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            CPClag7EIF_preds[column_name][index] = y_pred_eif[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# In[325]:


print(round(CPClag7EIF_preds["repeat2_fold1_timesplit5"].max(),2))


# ### 9D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[326]:


CPClag7EIF_preds[CPClag7EIF_preds["repeat2_fold2_timesplit3"]<=1].head()


# ### 9D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[327]:


CPClag7EIF_preds["anomaly"] = CPClag7EIF_preds.apply(lambda x: anomaly_result_LAG7CPC(x), axis=1)
CPClag7EIF_preds.head()


# In[328]:


CPClag7EIF_preds["anomaly"].max()


# ### 9D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.

# In[329]:


CPClag7EIF_preds[CPClag7EIF_preds["anomaly"]>=(CPClag7EIF_preds["anomaly"].max()-0.2)]["campaignId"].unique()


# ### 9D5 column containing percentage of less or equal to threshold per column in split + column containing anomaly score = save into csv

# In[330]:


# 1.79 is the max - 0.2
def anomaly_percent_EIF_LAG7CPC(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","CPC_lag7","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data >= 1.59:  #higher scores in the continuous distribution indicate anomaly
                negatives +=1
            elif data < 1.59:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    return negatives/(positives+negatives)


# In[331]:


CPClag7_eif = CPClag7EIF_preds.copy(deep=True)
CPClag7_eif["anom_percent"] = CPClag7_eif.apply(lambda x: anomaly_percent_EIF_LAG7CPC(x), axis=1)
CPClag7_eif.head(3)


# ### 9D6 save to csv based on date and ID, only anomaly and anom score

# In[332]:


eifcolsCPC7 = CPClag7_eif[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
eifcolsCPC7.columns = ['date','campaignId','anomaly_CPClag7_eif','anom_percent_CPClag7_eif']
eifcolsCPC7.head(3)


# In[ ]:


####


# ## Base original features dataframe
# #### Chapter 10A 

# In[333]:


base = clean.copy(deep=True)
base.head(2)


# ## Base features model application
# ### Chapter 10A1 Isolation Forest on base features
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[334]:


# https://datascience.stackexchange.com/a/78426
import math
from sklearn.ensemble import IsolationForest
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
base_preds = base.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = base["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(base['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(base.values):
        df_train, df_test = base.iloc[train_index], base.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Isolation Forest - new indices train and evaluation")
        isof = IsolationForest(random_state=123, max_samples='auto', contamination=0.01, 
                       max_features=1.0, bootstrap=False, n_jobs=-1, verbose=0).fit(df_train_final)
        y_pred_isof = isof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        base_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            base_preds[column_name][index] = y_pred_isof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 10A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[335]:


base_preds[base_preds["repeat2_fold2_timesplit3"]==-1].head()


# ### 10A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[336]:


def anomaly_result_base(row):     
    row2 = row.drop(["date", "campaignId", "biddingStrategyType", "cost", "portfolioBudget", "impressions", "clicks"])          
    result = 0
    for (name, data) in row2.iteritems():     
        timesplit = int(name[23:])         
        if data is not None:             
            column_value = timesplit * 0.2 * data         
        else:             
            column_value = 0
        result += column_value
    return round(result,2)


# In[337]:


base_preds["anomaly"] = base_preds.apply(lambda x: anomaly_result_base(x), axis=1)
base_preds.head()


# ### 10A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[338]:


base_preds[base_preds["anomaly"]<=-2]["campaignId"].unique()


# ### 10A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# In[339]:


base_isof = base_preds.copy(deep=True)


# In[340]:


def anomaly_percent_base(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","clicks", "impressions","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data == -1:
                negatives +=1
            elif data == 1:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    
    return negatives/(positives+negatives)


# In[341]:


base_isof["anom_percent"] = base_isof.apply(lambda x: anomaly_percent_base(x), axis=1)
base_isof.head()


# Value of 1 indicates 100% anomaly score.

# In[342]:


base_isof[base_isof["anom_percent"]==1]['campaignId'].unique()


# ### 10D6 save to csv based on date and ID, only anomaly and anom score

# In[343]:


isofcolsbase = base_isof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
isofcolsbase.columns = ['date','campaignId','anomaly_base_isof','anom_percent_base_isof']
isofcolsbase.head(3)


# In[750]:


####


# ### Chapter 10B1 Local Outlier Factor on base features
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[344]:


import math
from sklearn.neighbors import LocalOutlierFactor
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
base_predslof = base.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = base["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(base['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(base.values):
        df_train, df_test = base.iloc[train_index], base.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Local Outlier Factor - new indices train and evaluation")
        lof = LocalOutlierFactor(n_neighbors=20,novelty=False).fit_predict(df_train_final,df_test_final)
        #y_pred_lof = lof.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        base_predslof[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            base_predslof[column_name][index] = lof[own_index]#y_pred_lof[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 10B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[345]:


base_predslof[base_predslof["repeat2_fold2_timesplit3"]==-1]


# ### 10B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[346]:


base_predslof["anomaly"] = base_predslof.apply(lambda x: anomaly_result_base(x), axis=1)
base_predslof.head()


# ### 10B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[347]:


# which exact campaigns have anomaly score lower than or equal to -2 
base_predslof[base_predslof["anomaly"]<=-2.]["campaignId"].unique()


# In[348]:


base_lof = base_predslof.copy(deep=True)
base_lof["anom_percent"] = base_lof.apply(lambda x: anomaly_percent_base(x), axis=1)
base_lof.head()


# ### 10B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[349]:


base_lof[base_lof["anom_percent"]==1]['campaignId'].unique()


# ### 10B6 save to csv based on date and ID, only anomaly and anom score

# In[350]:


basecolslof = base_lof[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
basecolslof.columns = ['date','campaignId','anomaly_base_lof','anom_percent_base_lof']
basecolslof.head(2)


# In[ ]:


####


# ### Chapter 11C1 Elliptic Ellipse on base features
# ### Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[351]:


import math
from sklearn.covariance import EllipticEnvelope
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
baseEE_preds = base.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = base["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(base['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(base.values):
        df_train, df_test = base.iloc[train_index], base.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Elliptic Envelope - new indices train and evaluation")
        ee = EllipticEnvelope(random_state=0,support_fraction=1.0).fit(df_train_final)
        y_pred_ee = ee.predict(df_test_final)
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        baseEE_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            baseEE_preds[column_name][index] = y_pred_ee[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# ### 11C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[352]:


baseEE_preds[baseEE_preds["repeat2_fold2_timesplit3"]==-1]


# ### 11C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[353]:


baseEE_preds["anomaly"] = baseEE_preds.apply(lambda x: anomaly_result_base(x), axis=1)
baseEE_preds.head()


# ### 11C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.

# In[354]:


# which exact campaigns have anomaly score lower than or equal to -2 
baseEE_preds[baseEE_preds["anomaly"]<=-2.]["campaignId"].unique()


# In[355]:


base_ee = baseEE_preds.copy(deep=True)
base_ee["anom_percent"] = base_ee.apply(lambda x: anomaly_percent_base(x), axis=1)
base_ee.head()


# ### 11C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

# Value of 1 indicates 100% anomaly score.

# In[356]:


base_ee[base_ee["anom_percent"]==1]['campaignId'].unique()


# ### 11C6 save to csv based on date and ID, only anomaly and anom score

# In[357]:


basecolsee = base_ee[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
basecolsee.columns = ['date','campaignId','anomaly_base_ee','anom_percent_base_ee']
basecolsee.head(2)


# In[ ]:


####


# ### Base features Extended Isolation Forest
# 
# ### 11D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 

# In[358]:


import math
import eif as iso
from numpy.random import seed

seed(1)
total_splits = 3
total_repeats = 2
total_timesplits = 5

current_fold = 1

# Create a dataframe to contain columns with anomalies
baseEIF_preds = base.copy(deep=True)

# Set unique count of campaigns
unique_campaigns_ids = base["campaignId"].unique()

rkf = RepeatedKFold(n_splits=total_splits, n_repeats=total_repeats, random_state=42)

# Split by campaignId
for train_cust, test_cust in rkf.split(base['campaignId'].unique()):
    #print("training/testing with customers : " + str(train_cust)+"/"+str(test_cust))
   
    current_timesplit = 2
   
    train_cust = unique_campaigns_ids[train_cust]
    test_cust = unique_campaigns_ids[test_cust]
   
    # Split by date
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=total_timesplits)
    for train_index, test_index in tscv.split(base.values):
        df_train, df_test = base.iloc[train_index], base.iloc[test_index]
       
        # Keep the right campaigns for training/testing
        df_train_final = pd.concat([ df_train.groupby('campaignId').get_group(i) for i in train_cust ])
        df_test_final = pd.concat([ df_test.groupby('campaignId').get_group(i) for i in test_cust ])
       
        # Drop redundant columns here
        df_train_final.drop(["date"], axis=1, inplace=True)
        df_test_final.drop(["date"], axis=1, inplace=True)
        
        # Normalizing the quantitative values
        df_train_final.iloc[:,3:] = minmax.fit_transform(df_train_final.iloc[:,3:])
        df_test_final.iloc[:,3:] = minmax.transform(df_test_final.iloc[:,3:])
       
        # Initiate the model to fit and predict
        print("Extended Isolation Forest - new indices train and evaluation")
        eif = iso.iForest(df_train_final.to_numpy(),ntrees=100, sample_size=512, ExtensionLevel=1)
        y_pred_eif = eif.compute_paths(df_test_final.to_numpy())
       
        # Create column based on repeat, fold and timesplit index
        column_name = f"repeat{math.ceil(current_fold / total_splits)}_fold{(current_fold-1) % total_splits + 1}_timesplit{current_timesplit}"
        baseEIF_preds[column_name] = None
           
        own_index = 0
        for index, value in df_test_final.iterrows():
            baseEIF_preds[column_name][index] = y_pred_eif[own_index]
            own_index += 1
    
        current_timesplit += 1
   
    current_fold += 1


# In[359]:


print(round(baseEIF_preds["repeat2_fold1_timesplit5"].max(),2))


# ### 11D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.

# In[360]:


baseEIF_preds[baseEIF_preds["repeat2_fold2_timesplit3"]<=1]


# ### 11D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign

# In[361]:


baseEIF_preds["anomaly"] = baseEIF_preds.apply(lambda x: anomaly_result_base(x), axis=1)
baseEIF_preds.head()


# ### 11D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.

# In[362]:


baseEIF_preds["anomaly"].max()


# In[363]:


baseEIF_preds[baseEIF_preds["anomaly"]>=(baseEIF_preds["anomaly"].max()-0.2)]["campaignId"].unique()


# In[364]:


# 1.93 is the max - 0.2
def anomaly_percent_EIF_base(row):
    row2 = row.drop(["date","campaignId","biddingStrategyType","cost","portfolioBudget","clicks", "impressions","anomaly"])
    positives = 0
    negatives = 0
    
    for (name, data) in row2.iteritems():
        if data is not None:
            if data >= 1.73:  #higher scores in the continuous distribution indicate anomaly
                negatives +=1
            elif data < 1.73:
                positives +=1
    if (positives + negatives) == 0:
        return 0
    return negatives/(positives+negatives)


# ### 11D5 column containing percentage of less or equal to threshold per column in split + column containing anomaly score = save into csv

# In[365]:


baseeif = baseEIF_preds.copy(deep=True)
baseeif["anom_percent"] = baseeif.apply(lambda x: anomaly_percent_EIF_base(x), axis=1)
baseeif.head(3)


# ### 11D6 save to csv based on date and ID, only anomaly and anom score

# In[366]:


eifcolsbase = baseeif[['date','campaignId','anomaly','anom_percent']].copy(deep=True)
eifcolsbase.columns = ['date','campaignId','anomaly_base_eif','anom_percent_base_eif']
eifcolsbase.head(3)


# In[ ]:


####


# # All model anomaly scores concatenated

# In[367]:


# List of all model results
anoms = [CTR_isof,lofcols,eecols,eifcols,CTRlag1isof,CTRlag1lof,CTRlag1eecols,eifcolsCTR1,isofcolsCTR7,CTRlag7colslof,CTRlag7colsee,eifcolsCTR7,CPCcols,CPCcolslof,CPCcolsee,eifcolsCPC,isofcolsCPC1,CPClag1colslof,CPClag1colsee,eifcolsCPC1,CPClag7colslof,CPClag7colsee,isofcolsCPC7,eifcolsCPC7,isofcolsbase,basecolslof,basecolsee,eifcolsbase]
len(anoms)


# In[368]:


# Concatening all the models results from the list

# https://stackoverflow.com/a/52223263/14503062
#from functools import reduce

allmodels = reduce(lambda x,y: pd.merge(x,y, on=['date','campaignId'], how='outer'), anoms)
allmodels


# In[369]:


len(allmodels.campaignId.unique())


# In[370]:


allmodels["anomaly_CTR_isof"].value_counts()


# In[371]:


# dropping the percentage column as it is not used for the RQ
allanoms = allmodels[allmodels.columns.drop(list(allmodels.filter(regex='percent')))]
allanoms.head()


# In[372]:


# saving the resulting df
allanoms.to_csv("allanomalies2.csv")


# In[373]:


allanoms[allanoms['campaignId']==7941930]


# In[374]:


# Add the name of the first column 'index' to the csv file
#allanoms = pd.read_csv("allanomalies.csv", index_col="index")
grouped = pd.DataFrame(allanoms["campaignId"].unique(), columns=["campaignId"])

# Select all columns with predictions
models = allanoms.columns[2:].values

for model in models:
    
    # Group the data by campaign and current model
    dc = allanoms.groupby(["campaignId", model])["date"].count().reset_index()
    dc.columns = ["campaignId", model, "count"]

    # Use different conditions for the Extended Isolation Forest
    if "_eif" not in model:
        conditions = [(dc[model] <= -2) & (dc["count"] >= 5)]
    else:
        conditions = [(dc[model] >= 1.6) & (dc["count"] >= 5)]
        
    choices = [1]
    
    # Apply conditions to determine anomalies
    dc["result"] = np.select(conditions, choices, default=0)
    grouped_result = dc.groupby(["campaignId"])["result"].max().reset_index()
    
    # Create anomaly score for the current model grouped by campaign
    grouped[model] = grouped_result["result"]


# In[375]:


len(grouped.columns[1:])


# In[376]:


# specialist labels
labels = pd.read_excel('Anomaly detection.xlsx') 
labels


# # Model evaluation

# In[377]:


# Actual labels
y_true = labels["Anomaly"].tolist()

# Creating a container for all model classification outputs
ypred = []
for mod in grouped.columns[1:]:
    #print(grouped[mod])
    ypred.append(grouped[mod].tolist())


# ### accuracy

# In[378]:


from sklearn.metrics import accuracy_score

# Creating a container for all accuracies
acc = []
for preds in ypred:
    acc.append(accuracy_score(y_true,preds))
len(acc)


# ### computing weights and balanced accuracy

# In[379]:


from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
balweights = compute_sample_weight(class_weight= "balanced", y=y_true)

# Creating a container for all balanced accuracies
bal_acc = []

for balpreds in ypred:
    bal_acc.append(balanced_accuracy_score(y_true, balpreds, sample_weight=balweights))
len(bal_acc)


# In[380]:


from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
dictweights = compute_sample_weight(class_weight= {0:1,1:100}, y=y_true)

# Creating a container for all balanced accuracies
bal_acc2 = []

for balpreds in ypred:
    bal_acc2.append(balanced_accuracy_score(y_true, balpreds, sample_weight=dictweights))
len(bal_acc2)


# In[381]:


models = []
for name in grouped.columns[1:]:
    models.append(str(name))
len(models)


# In[382]:


resultdf = pd.DataFrame({"Model":models,
                         "Accuracy":acc,
                         "Balanced Accuracy":bal_acc,
                         "Balanced Weighted Accuracy":bal_acc2})
resultdf


# ## Confusion matrices of all models and dataframes

# In[383]:


# confusion matrix without weights
from sklearn.metrics import confusion_matrix
cm_now = []
for preds in ypred:
    cm_now.append(confusion_matrix(y_true, preds))
cm_now


# In[384]:


# confusion matrix with balanced weights
from sklearn.metrics import confusion_matrix
cm_bw = []
for preds in ypred:
    cm_bw.append(confusion_matrix(y_true, preds, sample_weight=balweights))
cm_bw


# In[385]:


# confusion matrix with balanced weighted weights
from sklearn.metrics import confusion_matrix
cm_bww = []
for preds in ypred:
    cm_bww.append(confusion_matrix(y_true, preds, sample_weight=dictweights))
cm_bww


# ## Confusion matrix of top performing model and dataframe

# ### CTR lag7 lof confusion matrix based on highest balanced and weighted balanced accuracy

# In[386]:


# Extracting the CM of balanced accuracy for best performing model
cm_now[9],cm_bw[9],cm_bww[9], # CTR lag7 LOF acc, bal_acc, bal_acc2
# accuracy not the highest, but the weighted ones are


# In[387]:


# Extracting the CM of accuracy for best performing model
cm_now[11],cm_bw[11],cm_bww[11] # CPC lag7 EIF acc, bal_acc, bal_acc2
# accuracy not the highest, but the weighted ones are


# In[388]:


# CTR lag7 lof confusion matrix
nowtf_CTRlag7lof = pd.DataFrame(data=cm_now[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwtf_CTRlag7lof = pd.DataFrame(data=cm_bw[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwwtf_CTRlag7lof = pd.DataFrame(data=cm_bww[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])

print("CTR Lag 7 LOF confusion matrix")
print("================================================================")
print("No weights Accuracy")
print("--------------------")
print(nowtf_CTRlag7lof)
print("================================================================")
print("Balanced weights Accuracy")
print("--------------------")
print(bwtf_CTRlag7lof)
print("================================================================")
print("Weighted balanced weights Accuracy")
print("--------------------")
print(bwwtf_CTRlag7lof)


# In[389]:


# CTR lag7 lof confusion matrix from bal.weight.acc.
tfdf_CTRlag7lof = pd.DataFrame(data=cm_bww[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
print(tfdf_CTRlag7lof)


# In[390]:


# CTR lag7 eif confusion matrix from acc.
tfdf_CTRlag7eif = pd.DataFrame(data=cm_now[11], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
print(tfdf_CTRlag7eif)


# In[391]:


# CTR lag7 eif confusion matrix
nowtf_CTRlag7eif = pd.DataFrame(data=cm_now[11], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwtf_CTRlag7eif = pd.DataFrame(data=cm_bw[11], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwwtf_CTRlag7eif = pd.DataFrame(data=cm_bww[11], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])

print("CTR Lag 7 EIF confusion matrix")
print("================================================================")
print("No weights Accuracy")
print("--------------------")
print(nowtf_CTRlag7eif)
print("================================================================")
print("Balanced weights Accuracy")
print("--------------------")
print(bwtf_CTRlag7eif)
print("================================================================")
print("Weighted balanced weights Accuracy")
print("--------------------")
print(bwwtf_CTRlag7eif)


# In[ ]:





# In[ ]:





# ## ROC curve 

# In[ ]:


### CTR lag 7 EIF


# In[392]:


from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped["anomaly_CTRlag7_eif"].tolist())

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('CTRlag7, EIF, Weightless',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)
#figure.show()


# In[393]:


from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped["anomaly_CTRlag7_eif"].tolist(),sample_weight=balweights)

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('CTRlag7, EIF, Balanced',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)
#figure.show()


# In[394]:


from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped["anomaly_CTRlag7_eif"].tolist(),sample_weight=dictweights)

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('CTRlag7, EIF, Weighted',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)
#figure.show()


# In[ ]:


### CTR lag 7 LOF


# In[395]:


from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped["anomaly_CTRlag7_lof"].tolist())

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('CTRlag7, LOF, Weightless',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)
#figure.show()


# In[396]:


from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped["anomaly_CTRlag7_lof"].tolist(),sample_weight=balweights)

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('CTRlag7, LOF, Balanced',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)
#figure.show()


# In[397]:


from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped["anomaly_CTRlag7_lof"].tolist(),sample_weight=dictweights)

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('CTRlag7, LOF, Weighted',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)
#figure.show()


# In[ ]:





# In[ ]:




