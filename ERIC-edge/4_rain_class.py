# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:53:42 2021

@author: Tian
"""

#------------------------------------------------------------------------
def cal_score(cm, method, mode):
   
    print(mode)
    #print(cm)

    TP=cm[1][1]
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    
    print("TP = ", TP)
    print("TN = ", TN)
    print("FP = ", FP)
    print("FN = ", FN)
    
    acc=(TP+TN)/(TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1=2*TP/(2*TP+FP+FN)
    MCC=(TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

    print("precision = ", precision)
    print("recall = " , recall)
    print("accuracy = ", acc)
    print("F1 = ", F1)
    print("MCC = ", MCC)

    print("--"*20)
    
    if mode == 'val':
       fw = fw_val
    elif mode == 'test':
       fw = fw_test
    
    fw.write('%15s' % method); fw.write("\t")
    fw.write('%15d' % TP); fw.write("\t")
    fw.write('%15d' % TN); fw.write("\t")
    fw.write('%15d' % FP); fw.write("\t")
    fw.write('%15d' % FN); fw.write("\t")
    fw.write('%15.4f' % precision); fw.write("\t")
    fw.write('%15.4f' % recall); fw.write("\t")
    fw.write('%15.4f' % acc); fw.write("\t")
    fw.write('%15.4f' % F1); fw.write("\t")
    fw.write('%15.4f' % MCC); 
    fw.write("\n")

#------------------------------------------------------------------------
# def cross_val(classifier, x_train, y_train):

#     if crossval:
#         print("10 fold-cross validation")
#         from sklearn.model_selection import cross_val_score
#         cv_score = cross_val_score(estimator=classifier, X=x_train,y=y_train, 
#                                    cv=10, scoring='f1_macro')
#         #cv_score = cross_val_score(estimator=classifier, X=x_train,y=y_train, cv=10)
#         print("cv_score_avg = ", cv_score.mean())
#         print("cv_score_std = ", cv_score.std())
#         print("--"*20)
    
#         fw.write('%15.4f' % cv_score.mean()); fw.write("\t")
#         fw.write('%15.4f' % cv_score.std())
#         fw.write("\n")
#     else:
#         fw.write("\n")
#         pass

def temporal_filter(y_pred):
    # temporal filter to remove false positives in ANN of single point
  result = []
  result.append(y_pred[0])
  for i in range(1, len(y_pred)-1):
    if y_pred[i]==1 and (y_pred[i-1] == 1 or y_pred[i+1] == 1): # temproal filter only when consecutive two labels are 1 / consecutive rain
      result.append(1)
    else:
      result.append(0)
  
  result.append(y_pred[-1])

  return result


#------------------------------------------------------------------------ main
import pandas as pd
import math
import yaml
import os

with open('control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_path = controls['io_path']
output_path = controls['io_path']

fea_suffix = controls['fea_suffix']
time_suffix = controls['time_suffix']

crossval = False
t_filter = True

if not os.path.isdir(output_path):
  os.mkdir(output_path)
  print("Folder {} created".format(output_path))
else:
  print("Folder {} already existed".format(output_path))
  
# create accuracy log file
fw_val = open(output_path + 'rain_detection_val_' + fea_suffix + time_suffix + '.log','w')
col_names = ['Model','TP','TN','FP','FN','Precision','Recall','Accuracy','F1 Score','MCC']

for name in col_names:
    fw_val.write('%15s' % name); fw_val.write("\t")
fw_val.write('\n')

fw_test = open(output_path + 'rain_detection_test_' + fea_suffix + time_suffix + '.log','w')
col_names = ['Model','TP','TN','FP','FN','Precision','Recall','Accuracy','F1 Score','MCC']

for name in col_names:
    fw_test.write('%15s' % name); fw_test.write("\t")
fw_test.write('\n')

# read in data
train = pd.read_csv(input_path + 'train.csv')
val = pd.read_csv(input_path + 'val.csv')
test = pd.read_csv(input_path + 'test.csv')

# select features to be used for ML
# fea_name_list = ['RMSE', 'AmplitudeEnvelope']
# data = dataset[fea_name_list]

if fea_suffix == 'audio-only':
  x_train = train.iloc[:,2:7]
  x_val = val.iloc[:,2:7]
  x_test = test.iloc[:,2:7]
  
elif fea_suffix == 'visual-only':
  x_train = train.iloc[:,7:27]
  x_val = val.iloc[:,7:27]
  x_test = test.iloc[:,7:27]
  
elif fea_suffix == 'audio-visual':
  # data = dataset.iloc[:,1:46] # include is_day, MFCCs
  # removed MFCCs, turn out 20 MFCCs introduces a lot of noises # remove is_day
  x_train = train.iloc[:,2:27]
  x_val = val.iloc[:,2:27]
  x_test = test.iloc[:,2:27]
  
else:
  print('Error! Unrecognized fea_suffix.')

x_train = x_train.values
x_val = x_val.values
x_test = x_test.values

y_train = train['is_rain'].values
y_val = val['is_rain'].values
y_test = test['is_rain'].values

#----------------- feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_val = sc_X.transform(x_val)
x_test = sc_X.transform(x_test)

# # save the scaler
# from pickle import dump
# dump(sc_X, open(output_path+'scaler.pkl', 'wb'))

# shuffle the train data
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=100)

# val/test results for plotting is_rain
time_val = val['DateTime']
val_res = pd.DataFrame(time_val, columns=['DateTime'])
val_res['is_rain_true'] = y_val

time_test = test['DateTime']
test_res = pd.DataFrame(time_test, columns=['DateTime'])
test_res['is_rain_true'] = y_test

#----------------------- feature importance by correlation
"""
import seaborn as sns
import matplotlib.pyplot as plt

dataset = train.iloc[:,2:28] # vs. is_rain
corrmat = dataset.corr()
top_corr_features = corrmat.index

plt.figure()
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
"""

#----------------------------------------------------- logistic regression
method="LR"
print(method)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 100)
classifier.fit(x_train, y_train)

y_val_res = classifier.predict(x_val)
y_test_res = classifier.predict(x_test)

# apply temporal filtering
if t_filter:
  y_val_res = temporal_filter(y_val_res)
  y_test_res = temporal_filter(y_test_res)

# change threshold value
#y_pred_prob = classifier.predict_proba(x_test)
#decisions = (classifier.predict_proba(x_test)[:,1] >= 0.4).astype(int)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val,y_val_res)
cal_score(cm_val, method, 'val')

cm_test = confusion_matrix(y_test, y_test_res)
cal_score(cm_test, method, 'test')

val_res['LR'] = y_val_res
test_res['LR'] = y_test_res

# 10-Fold Cross Validation
# cross_val(classifier, x_train, y_train)

#----------------------------------------------------- KNN
"""
method = "KNN"
print(method)

#fitting the classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

y_val_res = classifier.predict(x_val)
y_test_res = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val,y_val_res)
cal_score(cm_val, method, 'val')

cm_test = confusion_matrix(y_test, y_test_res)
cal_score(cm_test, method, 'test')

val_res['KNN'] = y_val_res
test_res['KNN'] = y_test_res

# cross_val(classifier, x_train, y_train)
"""
#-----------------------------------------------------SVM
"""
method="SVM"
print(method)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=100)
classifier.fit(x_train, y_train)

y_val_res = classifier.predict(x_val)
y_test_res = classifier.predict(x_test)

# apply temporal filtering
if t_filter:
  y_val_res = temporal_filter(y_val_res)
  y_test_res = temporal_filter(y_test_res)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val,y_val_res)
cal_score(cm_val, method, 'val')

cm_test = confusion_matrix(y_test, y_test_res)
cal_score(cm_test, method, 'test')

val_res['SVM'] = y_val_res
test_res['SVM'] = y_test_res

# cross_val(classifier, x_train, y_train)
"""

#-----------------------------------------------------Naive Bayes

method = "NaiveBayes"
print(method)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_val_res = classifier.predict(x_val)
y_test_res = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val,y_val_res)
cal_score(cm_val, method, 'val')

cm_test = confusion_matrix(y_test, y_test_res)
cal_score(cm_test, method, 'test')

val_res['NaiveBayes'] = y_val_res
test_res['NaiveBayes'] = y_test_res

# cross_val(classifier, x_train, y_train)

#-----------------------------------------------------Decision Tree
"""
method = "DecisionTree"
print(method)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=100)
classifier.fit(x_train,y_train)

y_val_res = classifier.predict(x_val)
y_test_res = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val,y_val_res)
cal_score(cm_val, method, 'val')

cm_test = confusion_matrix(y_test, y_test_res)
cal_score(cm_test, method, 'test')

val_res['DecisionTree'] = y_val_res
test_res['DecisionTree'] = y_test_res

# cross_val(classifier, x_train, y_train)
"""
#-----------------------------------------------------Random Forest
method = "RandomForest"
print(method)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=100)
classifier.fit(x_train,y_train)

y_val_res = classifier.predict(x_val)
y_test_res = classifier.predict(x_test)

# apply temporal filtering
if t_filter:
  y_val_res = temporal_filter(y_val_res)
  y_test_res = temporal_filter(y_test_res)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val,y_val_res)
cal_score(cm_val, method, 'val')

cm_test = confusion_matrix(y_test, y_test_res)
cal_score(cm_test, method, 'test')

val_res['RandomForest'] = y_val_res
test_res['RandomForest'] = y_test_res

# cross_val(classifier, x_train, y_train)

#----------------------------- tree-based feature importance
"""
import seaborn as sns
import matplotlib.pyplot as plt

data = train.copy(deep=True).iloc[:, 2:28]
# data.drop(['DateTime','is_day','rain_amount'], axis = 1, inplace = True)

feats = {}
for feature, importance in zip(data.columns, classifier.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})

sns.set(font_scale = 5)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.0)

fig, ax = plt.subplots()
fig.set_size_inches(5,4.5)
sns.barplot(y=importances['Gini-Importance'], x=importances['Features'], data=importances, color='skyblue')
plt.xticks(rotation = 45)
plt.ylim(0, 0.12)
plt.ylabel('Gini-Importance', fontsize=11, weight = 'bold')
# plt.xlabel('Features', fontsize=11, weight = 'bold')
#plt.title('Feature Importance', fontsize=25, weight = 'bold')
# plt.tight_layout()
plt.show()

print(importances)
"""

#-----------------------------------------------------XGBoost
"""
method = "XGBoost"
print(method)
from xgboost import XGBClassifier
classifier = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=100)
classifier.fit(x_train,y_train)

y_val_res = classifier.predict(x_val)
y_test_res = classifier.predict(x_test)

# apply temporal filtering
if t_filter:
  y_val_res = temporal_filter(y_val_res)
  y_test_res = temporal_filter(y_test_res)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val,y_val_res)
cal_score(cm_val, method, 'val')

cm_test = confusion_matrix(y_test, y_test_res)
cal_score(cm_test, method, 'test')

val_res['RandomForest'] = y_val_res
test_res['RandomForest'] = y_test_res

# cross_val(classifier, x_train, y_train)
"""

#-----------------------------------------------------ANN
method="ANN"
print(method)
from tensorflow import keras
from keras.models import Sequential #used to initialize the ANN
from keras.layers import Dense #used to add the different layers

#initialize the ANN
classifier=Sequential()
classifier.add(Dense(units=6, activation='relu',input_dim=x_train.shape[1])) #create first hidden layer

#add second hidden layer
classifier.add(Dense(units=6, activation='relu'))
#add the output layer
classifier.add(Dense(units=1, activation='sigmoid')) #get the probability of each customer leaving the bank 
#compile the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint(output_path+'ANN_detector_model.best', save_best_only=True, monitor='val_loss', mode='min')

classifier.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=30, callbacks=[mcp_save])

# save the trained NN model
# classifier.save(output_path+'ANN_class.model')
classifier = keras.models.load_model(output_path+'ANN_detector_model.best')

y_val_res = classifier.predict(x_val)
y_val_res = (y_val_res>0.5)
y_val_res = [int(res) for res in y_val_res]

y_test_res = classifier.predict(x_test)
y_test_res = (y_test_res>0.5)
y_test_res = [int(res) for res in y_test_res]

# apply temporal filtering
if t_filter:
  y_val_res = temporal_filter(y_val_res)
  y_test_res = temporal_filter(y_test_res)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val, y_val_res)
cal_score(cm_val, method, 'val')

cm_test = confusion_matrix(y_test, y_test_res)
cal_score(cm_test, method, 'test')

val_res['ANN'] = y_val_res
test_res['ANN'] = y_test_res

#-------------------------------------------------------- end
fw_val.close()
fw_test.close()

val_res.to_csv(output_path + 'israin_val.csv', index=False)
test_res.to_csv(output_path + 'israin_test.csv', index=False)
