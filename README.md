# Effective Feature Representations for Anomaly Detection in Search Advertising Time Series
Tilburg University Data Science & Society MSc Thesis

Below are the structured comments directly found within the python file to help navigate the thesis code structure:

### Models:

### A: Isolation Forest 
4A1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
4A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
4A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
4A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
4A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### B: Local Outlier Factor
4B1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. LOF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
4B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
4B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
4B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
4B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### C: Elliptic Envelope
4C1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EE model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
4C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
4C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
4C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
4C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

D: Extended Isolation Forest
4D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
Anomalies are those that occur less frequently, hence, the number of points with higher anomaly scores reduces as the score increases
4D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
4D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
4D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.
4D5 column containing percentage of anomaly per column in split + column containing anomaly score = save into csv

### 5. Creating feature representations
### CTR lag 1 dataframe
### CTR lag 1 model application

Chapter 5A1 Isolation Forest on CTR lag 1

Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
5A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
5A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
5A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
5A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

Chapter 5B1 Local Outlier Factor on CTR lag 1

Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
5B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
5B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
5B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
5B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

Chapter 5C1 Elliptic Ellipse on CTR lag 1
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
5C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
5C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
5C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
5C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

CTR Lag 1 Extended Isolation Forest
5D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
5D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
5D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
5D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.
5D5 column containing percentage per column in split + column containing anomaly score = save into csv

### CTR lag 7 dataframe
### CTR lag 7 model application
### Chapter 6A1 Isolation Forest on CTR lag 7
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
6A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
6A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
6A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
6A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 6B1 Local Outlier Factor on CTR lag 7
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
6B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
6B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
6B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
6B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 6C1 Elliptic Ellipse on CTR lag 7
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
6C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
6C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
6C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
6C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv
6C6 save to csv based on date and ID, only anomaly and anom score

### CTR Lag 1 Extended Isolation Forest
6D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
6D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
6D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
6D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.
6D5 column containing percentage of lower than (max-0.2) per column in split + column containing anomaly score = save into csv

### CPC dataframe
### CPC model application
### Chapter 7A1 Isolation Forest on CPC
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
7A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
7A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
7A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
7A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 7B1 Local Outlier Factor on CPC
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
7B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
7B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
7B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
7B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 7C1 Elliptic Ellipse on CPC
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
7C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
7C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
7C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
7C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### CPC Extended Isolation Forest
7D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
7D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
7D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
7D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.
7D5 column containing percentage of less or equal to threshold per column in split + column containing anomaly score = save into csv

### CPC lag 1 dataframe
### CPC lag 1model application
### Chapter 8A1 Isolation Forest on CPC lag 1
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set.
8A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
8A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
8A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
8A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 8B1 Local Outlier Factor on CPC lag 1 
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
8B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
8B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
8B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
8B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 8C1 Elliptic Ellipse on CPC lag 1
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
8C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
8C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
8C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
8C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv
8C6 save to csv based on date and ID, only anomaly and anom score

### CPC Lag 1 Extended Isolation Forest
8D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
8D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
8D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
8D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.
8D5 column containing percentage of less or equal to threshold per column in split + column containing anomaly score = save into csv

### CPC lag 7 dataframe
### CPC lag 7 model application
### Chapter 9A1 Isolation Forest on CPC lag 7
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
9A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
9A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
9A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
9A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 9B1 Local Outlier Factor on CPC lag 7
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
9B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
9B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
9B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
9B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 9C1 Elliptic Ellipse on CPC lag 7
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
9C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
9C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
9C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
9C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### CPC Lag 7 Extended Isolation Forest
9D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
9D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
9D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
9D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.
9D5 column containing percentage of less or equal to threshold per column in split + column containing anomaly score = save into csv

### Base original features dataframe
### Base features model application
### Chapter 10A1 Isolation Forest on base features
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
10A2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
10A3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
10A4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
10A5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 10B1 Local Outlier Factor on base features
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
10B2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
10B3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
10B4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
10B5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Chapter 11C1 Elliptic Ellipse on base features
Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. Isolation Forest model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
11C2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
11C3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
11C4 From unique values of the anomaly score, x <= -2 is taken as detected anomaly, as those had the two highest weights and two lowest calculated score.
11C5 column containing percentage of -1 per column in split + column containing anomaly score = save into csv

### Base features Extended Isolation Forest
11D1 Data is split by k=3 fold twice, and then split by time 5 times, creating 30 separate splits based on date and campaignId. EIF model is applied on each split, creating a column with -1 where an anomaly is encountered, and None where the row was used in the test set. 
11D2 Each split indicates whether the model saw it as an anomaly, hence an average needs to be taken.
11D3 Custom function to take a weighted average from the splits. Later splits have more importance as they take more data into account and hence have more weight. A new column is assigned the calculated function result, from which the lowest values are the highest possibility of an anomalous campaign
11D4 From unique values of the anomaly score, x >= (max-0.2) is taken as detected anomaly, as those had the highest weights and lowest calculated score.
11D5 column containing percentage of less or equal to threshold per column in split + column containing anomaly score = save into csv

### All model anomaly scores concatenated

### Model evaluation
Creating a container for all model classification outputs
### Accuracy
Creating a container for all accuracies
computing weights and balanced accuracy
Creating balanced sample weights based on actual labels
Creating a container for all balanced accuracies
### Confusion matrices of all models and dataframes
Confusion matrix without weights
Confusion matrix with balanced weights
Confusion matrix with balanced weighted weights

### Confusion matrix of top performing model and dataframe
CTR lag7 lof confusion matrix based on highest balanced and weighted balanced accuracy
Extracting the CM of balanced accuracy for best performing model
Accuracy not the highest, but the weighted ones are

### CTR lag7 lof confusion matrix
### CTR lag7 lof confusion matrix from bal.weight.acc.
### CTR lag7 eif confusion matrix from acc.
### CTR lag7 eif confusion matrix

### ROC curve 
### CTR lag 7 EIF
### CTR lag 7 LOF





