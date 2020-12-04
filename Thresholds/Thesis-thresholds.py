#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[5]:


allanoms = pd.read_csv("allanomalies2.csv",index_col=0)
allanoms.head()


# In[7]:


# ### test with 4 as count threshold


# Add the name of the first column 'index' to the csv file
#allanoms = pd.read_csv("allanomalies.csv", index_col="index")
grouped4 = pd.DataFrame(allanoms["campaignId"].unique(), columns=["campaignId"])

# Select all columns with predictions
models = allanoms.columns[2:].values

for model in models:
    
    # Group the data by campaign and current model
    dc = allanoms.groupby(["campaignId", model])["date"].count().reset_index()
    dc.columns = ["campaignId", model, "count"]

    # Use different conditions for the Extended Isolation Forest
    if "_eif" not in model:
        conditions = [(dc[model] <= -2) & (dc["count"] >= 3)]
    else:
        conditions = [(dc[model] >= 1.6) & (dc["count"] >= 3)]
        
    choices = [1]
    
    # Apply conditions to determine anomalies
    dc["result"] = np.select(conditions, choices, default=0)
    grouped_result = dc.groupby(["campaignId"])["result"].max().reset_index()
    
    # Create anomaly score for the current model grouped by campaign
    grouped4[model] = grouped_result["result"]


# In[8]:


# ### test with 3 as count threshold


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
        conditions = [(dc[model] <= -2) & (dc["count"] >= 3)]
    else:
        conditions = [(dc[model] >= 1.6) & (dc["count"] >= 3)]
        
    choices = [1]
    
    # Apply conditions to determine anomalies
    dc["result"] = np.select(conditions, choices, default=0)
    grouped_result = dc.groupby(["campaignId"])["result"].max().reset_index()
    
    # Create anomaly score for the current model grouped by campaign
    grouped[model] = grouped_result["result"]


# In[9]:


# ### test with 2 as count threshold


# Add the name of the first column 'index' to the csv file
#allanoms = pd.read_csv("allanomalies.csv", index_col="index")
grouped2 = pd.DataFrame(allanoms["campaignId"].unique(), columns=["campaignId"])

# Select all columns with predictions
models = allanoms.columns[2:].values

for model in models:
    
    # Group the data by campaign and current model
    dc = allanoms.groupby(["campaignId", model])["date"].count().reset_index()
    dc.columns = ["campaignId", model, "count"]

    # Use different conditions for the Extended Isolation Forest
    if "_eif" not in model:
        conditions = [(dc[model] <= -2) & (dc["count"] >= 2)]
    else:
        conditions = [(dc[model] >= 1.6) & (dc["count"] >= 2)]
        
    choices = [1]
    
    # Apply conditions to determine anomalies
    dc["result"] = np.select(conditions, choices, default=0)
    grouped_result = dc.groupby(["campaignId"])["result"].max().reset_index()
    
    # Create anomaly score for the current model grouped by campaign
    grouped2[model] = grouped_result["result"]


# In[10]:


# ### test with 1 as count threshold


# Add the name of the first column 'index' to the csv file
#allanoms = pd.read_csv("allanomalies.csv", index_col="index")
grouped1 = pd.DataFrame(allanoms["campaignId"].unique(), columns=["campaignId"])

# Select all columns with predictions
models = allanoms.columns[2:].values

for model in models:
    
    # Group the data by campaign and current model
    dc = allanoms.groupby(["campaignId", model])["date"].count().reset_index()
    dc.columns = ["campaignId", model, "count"]

    # Use different conditions for the Extended Isolation Forest
    if "_eif" not in model:
        conditions = [(dc[model] <= -2) & (dc["count"] >= 1)]
    else:
        conditions = [(dc[model] >= 1.6) & (dc["count"] >= 1)]
        
    choices = [1]
    
    # Apply conditions to determine anomalies
    dc["result"] = np.select(conditions, choices, default=0)
    grouped_result = dc.groupby(["campaignId"])["result"].max().reset_index()
    
    # Create anomaly score for the current model grouped by campaign
    grouped1[model] = grouped_result["result"]


# In[11]:


# ### ground truth

labels = pd.read_excel('Anomaly detection.xlsx') 
labels


# In[12]:


# # Model evaluation

### 4 as threshold

# Actual labels
y_true = labels["Anomaly"].tolist()

# Creating a container for all model classification outputs
ypred4 = []
for mod in grouped4.columns[1:]:
    #print(grouped[mod])
    ypred4.append(grouped4[mod].tolist())


# In[13]:


### 3 as threshold

# Actual labels
y_true = labels["Anomaly"].tolist()

# Creating a container for all model classification outputs
ypred3 = []
for mod in grouped.columns[1:]:
    #print(grouped[mod])
    ypred3.append(grouped[mod].tolist())


# In[14]:


### 2 as threshold

# Actual labels
y_true = labels["Anomaly"].tolist()

# Creating a container for all model classification outputs
ypred2 = []
for mod in grouped2.columns[1:]:
    #print(grouped[mod])
    ypred2.append(grouped2[mod].tolist())


# In[15]:


### 1 as threshold

# Actual labels
y_true = labels["Anomaly"].tolist()

# Creating a container for all model classification outputs
ypred1 = []
for mod in grouped1.columns[1:]:
    #print(grouped[mod])
    ypred1.append(grouped1[mod].tolist())


# In[ ]:


### accuracy


# In[16]:


### 4 as threshold

from sklearn.metrics import accuracy_score

# Creating a container for all accuracies
acc4 = []
for preds in ypred4:
    acc4.append(accuracy_score(y_true,preds))
len(acc4)


# In[18]:


### 3 as threshold

from sklearn.metrics import accuracy_score

# Creating a container for all accuracies
acc3 = []
for preds in ypred3:
    acc3.append(accuracy_score(y_true,preds))
len(acc3)


# In[19]:


### 2 as threshold

from sklearn.metrics import accuracy_score

# Creating a container for all accuracies
acc2 = []
for preds in ypred2:
    acc2.append(accuracy_score(y_true,preds))
len(acc2)


# In[20]:


### 1 as threshold

from sklearn.metrics import accuracy_score

# Creating a container for all accuracies
acc1 = []
for preds in ypred1:
    acc1.append(accuracy_score(y_true,preds))
len(acc1)


# ### computing weights and balanced accuracy

# In[21]:


### 4 as threshold

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
balweights = compute_sample_weight(class_weight= "balanced", y=y_true)

# Creating a container for all balanced accuracies
bal_acc4 = []

for balpreds in ypred4:
    bal_acc4.append(balanced_accuracy_score(y_true, balpreds, sample_weight=balweights))
len(bal_acc4)


# In[22]:


### 3 as threshold

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
balweights3 = compute_sample_weight(class_weight= "balanced", y=y_true)

# Creating a container for all balanced accuracies
bal_acc3 = []

for balpreds in ypred3:
    bal_acc3.append(balanced_accuracy_score(y_true, balpreds, sample_weight=balweights3))
len(bal_acc3)


# In[23]:


### 2 as threshold

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
balweights2 = compute_sample_weight(class_weight= "balanced", y=y_true)

# Creating a container for all balanced accuracies
bal_acc2 = []

for balpreds in ypred2:
    bal_acc2.append(balanced_accuracy_score(y_true, balpreds, sample_weight=balweights2))
len(bal_acc2)


# In[24]:


### 1 as threshold

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
balweights1 = compute_sample_weight(class_weight= "balanced", y=y_true)

# Creating a container for all balanced accuracies
bal_acc1 = []

for balpreds in ypred1:
    bal_acc1.append(balanced_accuracy_score(y_true, balpreds, sample_weight=balweights1))
len(bal_acc1)


# In[25]:


### 4 as threshold

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
dictweights4 = compute_sample_weight(class_weight= {0:1,1:100}, y=y_true)

# Creating a container for all balanced accuracies
bal_acc2_4 = []

for balpreds in ypred4:
    bal_acc2_4.append(balanced_accuracy_score(y_true, balpreds, sample_weight=dictweights4))
len(bal_acc2_4)


# In[26]:


### 3 as threshold

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
dictweights3 = compute_sample_weight(class_weight= {0:1,1:100}, y=y_true)

# Creating a container for all balanced accuracies
bal_acc2_3 = []

for balpreds in ypred3:
    bal_acc2_3.append(balanced_accuracy_score(y_true, balpreds, sample_weight=dictweights3))
len(bal_acc2_3)


# In[27]:


### 2 as threshold

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
dictweights2 = compute_sample_weight(class_weight= {0:1,1:100}, y=y_true)

# Creating a container for all balanced accuracies
bal_acc2_2 = []

for balpreds in ypred2:
    bal_acc2_2.append(balanced_accuracy_score(y_true, balpreds, sample_weight=dictweights2))
len(bal_acc2_2)


# In[28]:


### 1 as threshold

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Creating balanced sample weights based on actual labels
dictweights1 = compute_sample_weight(class_weight= {0:1,1:100}, y=y_true)

# Creating a container for all balanced accuracies
bal_acc2_1 = []

for balpreds in ypred1:
    bal_acc2_1.append(balanced_accuracy_score(y_true, balpreds, sample_weight=dictweights1))
len(bal_acc2_1)


# In[29]:


models4 = []
for name in grouped4.columns[1:]:
    models4.append(str(name))
len(models4)


# In[30]:


### 4 as threshold

resultdf4 = pd.DataFrame({"Model":models4,
                         "Accuracy":acc4,
                         "Balanced Accuracy":bal_acc4,
                         "Balanced Weighted Accuracy":bal_acc2_4})
resultdf4


# In[31]:


models3 = []
for name in grouped.columns[1:]:
    models3.append(str(name))
len(models3)


# In[32]:


### 3 as threshold

resultdf3 = pd.DataFrame({"Model":models3,
                         "Accuracy":acc3,
                         "Balanced Accuracy":bal_acc3,
                         "Balanced Weighted Accuracy":bal_acc2_3})
resultdf3


# In[33]:


models2 = []
for name in grouped2.columns[1:]:
    models2.append(str(name))
len(models2)


# In[34]:


## 2 as threshold

resultdf2 = pd.DataFrame({"Model":models2,
                         "Accuracy":acc2,
                         "Balanced Accuracy":bal_acc2,
                         "Balanced Weighted Accuracy":bal_acc2_2})
resultdf2


# In[35]:


models1 = []
for name in grouped1.columns[1:]:
    models1.append(str(name))
len(models1)


# In[36]:


## 1 as threshold

resultdf1 = pd.DataFrame({"Model":models1,
                         "Accuracy":acc1,
                         "Balanced Accuracy":bal_acc1,
                         "Balanced Weighted Accuracy":bal_acc2_1})
resultdf1


# ## Confusion matrices of all models and dataframes

# In[37]:


### 4 as threshold

# confusion matrix without weights
from sklearn.metrics import confusion_matrix
cm_now4 = []
for preds in ypred4:
    cm_now4.append(confusion_matrix(y_true, preds))


# In[38]:


### 4 as threshold

# confusion matrix with balanced weights
from sklearn.metrics import confusion_matrix
cm_bw4 = []
for preds in ypred4:
    cm_bw4.append(confusion_matrix(y_true, preds, sample_weight=balweights))


# In[39]:


### 4 as threshold

# confusion matrix with balanced weighted weights
from sklearn.metrics import confusion_matrix
cm_bww4 = []
for preds in ypred4:
    cm_bww4.append(confusion_matrix(y_true, preds, sample_weight=dictweights4))


# In[40]:


### 3 as threshold

# confusion matrix without weights
from sklearn.metrics import confusion_matrix
cm_now3 = []
for preds in ypred3:
    cm_now3.append(confusion_matrix(y_true, preds))


# In[41]:


### 3 as threshold

# confusion matrix with balanced weights
from sklearn.metrics import confusion_matrix
cm_bw3 = []
for preds in ypred3:
    cm_bw3.append(confusion_matrix(y_true, preds, sample_weight=balweights3))


# In[42]:


### 3 as threshold

# confusion matrix with balanced weighted weights
from sklearn.metrics import confusion_matrix
cm_bww3 = []
for preds in ypred3:
    cm_bww3.append(confusion_matrix(y_true, preds, sample_weight=dictweights3))


# In[43]:


### 2 as threshold

# confusion matrix without weights
from sklearn.metrics import confusion_matrix
cm_now2 = []
for preds in ypred2:
    cm_now2.append(confusion_matrix(y_true, preds))


# In[44]:


### 2 as threshold

# confusion matrix with balanced weights
from sklearn.metrics import confusion_matrix
cm_bw2 = []
for preds in ypred2:
    cm_bw2.append(confusion_matrix(y_true, preds, sample_weight=balweights2))


# In[45]:


### 2 as threshold

# confusion matrix with balanced weighted weights
from sklearn.metrics import confusion_matrix
cm_bww2 = []
for preds in ypred2:
    cm_bww2.append(confusion_matrix(y_true, preds, sample_weight=dictweights2))


# In[46]:


### 1 as threshold

# confusion matrix without weights
from sklearn.metrics import confusion_matrix
cm_now1 = []
for preds in ypred1:
    cm_now1.append(confusion_matrix(y_true, preds))


# In[47]:


### 1 as threshold

# confusion matrix with balanced weights
from sklearn.metrics import confusion_matrix
cm_bw1 = []
for preds in ypred1:
    cm_bw1.append(confusion_matrix(y_true, preds, sample_weight=balweights1))


# In[48]:


### 1 as threshold

# confusion matrix with balanced weighted weights
from sklearn.metrics import confusion_matrix
cm_bww1 = []
for preds in ypred1:
    cm_bww1.append(confusion_matrix(y_true, preds, sample_weight=dictweights1))


# ## Confusion matrix of top performing model and dataframe

# ### CTR lag7 lof confusion matrix based on highest balanced and weighted balanced accuracy

# In[51]:


## 4 as threshold

# CTR lag7 lof confusion matrix
nowtf_CTRlag7lof4 = pd.DataFrame(data=cm_now4[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwtf_CTRlag7lof4 = pd.DataFrame(data=cm_bw4[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwwtf_CTRlag7lof4 = pd.DataFrame(data=cm_bww4[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])

print("CTR Lag 7 LOF confusion matrix")
print("================================================================")
print("No weights Accuracy")
print("--------------------")
print(nowtf_CTRlag7lof4)
print("================================================================")
print("Balanced weights Accuracy")
print("--------------------")
print(bwtf_CTRlag7lof4)
print("================================================================")
print("Weighted balanced weights Accuracy")
print("--------------------")
print(bwwtf_CTRlag7lof4)


# In[53]:


## 3 as threshold

# CTR lag7 lof confusion matrix
nowtf_CTRlag7lof = pd.DataFrame(data=cm_now3[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwtf_CTRlag7lof = pd.DataFrame(data=cm_bw3[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwwtf_CTRlag7lof = pd.DataFrame(data=cm_bww3[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])

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


# In[54]:


## 2 as threshold

# CTR lag7 lof confusion matrix
nowtf_CTRlag7lof2 = pd.DataFrame(data=cm_now2[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwtf_CTRlag7lof2 = pd.DataFrame(data=cm_bw2[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwwtf_CTRlag7lof2 = pd.DataFrame(data=cm_bww2[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])

print("CTR Lag 7 LOF confusion matrix")
print("================================================================")
print("No weights Accuracy")
print("--------------------")
print(nowtf_CTRlag7lof2)
print("================================================================")
print("Balanced weights Accuracy")
print("--------------------")
print(bwtf_CTRlag7lof2)
print("================================================================")
print("Weighted balanced weights Accuracy")
print("--------------------")
print(bwwtf_CTRlag7lof2)


# In[55]:


## 1 as threshold

# CTR lag7 lof confusion matrix
nowtf_CTRlag7lof1 = pd.DataFrame(data=cm_now1[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwtf_CTRlag7lof1 = pd.DataFrame(data=cm_bw1[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])
bwwtf_CTRlag7lof1 = pd.DataFrame(data=cm_bww1[9], index=["Actual Positives (1)", "Actual Negatives (0)"], columns=["Predicted Positive (1)", "Predicted Negative (0)"])

print("CTR Lag 7 LOF confusion matrix")
print("================================================================")
print("No weights Accuracy")
print("--------------------")
print(nowtf_CTRlag7lof1)
print("================================================================")
print("Balanced weights Accuracy")
print("--------------------")
print(bwtf_CTRlag7lof1)
print("================================================================")
print("Weighted balanced weights Accuracy")
print("--------------------")
print(bwwtf_CTRlag7lof1)


# In[64]:


## 1 as threshold

from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped1["anomaly_CTRlag7_lof"].tolist())

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('Classifier',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)


# In[61]:


## 2 as threshold

from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped2["anomaly_CTRlag7_lof"].tolist())

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('Classifier',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)


# In[62]:


## 3 as threshold

from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped["anomaly_CTRlag7_lof"].tolist())

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('Classifier',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)


# In[63]:


## 4 as threshold

from sklearn.metrics import roc_curve, auc
figure, ax1 = plt.subplots(figsize=(8,8))

fpr,tpr,_ = roc_curve(labels["Anomaly"].tolist(),grouped4["anomaly_CTRlag7_lof"].tolist())

# compute AUC
roc_auc = auc(fpr,tpr)

# plotting FPR and TPR
ax1.plot(fpr,tpr, label='%s (area = %0.2f)' % ('Classifier',roc_auc))
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate', fontsize=18)
ax1.set_ylabel('True Positive Rate', fontsize=18)
ax1.set_title("Receiver Operating Characteristic", fontsize=18)
plt.tick_params(axis='both', labelsize=18)
ax1.legend(loc="lower right", fontsize=14)
plt.grid(True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




