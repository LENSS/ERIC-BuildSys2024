# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:09:23 2021

@author: Tian
"""

def fix_negative(fix_negval, y_pred):
   
   if fix_negval:
       for i in range(len(y_pred)):
          #  if y_pred[i] < 0.01:
           if y_pred[i] < 0:
               y_pred[i] = 0
      
   return y_pred 
   
def cal_score(y_test, y_pred, mode):
   
    print(mode)
    s1 = sum(y_test)
    s2 = sum(y_pred)
    error_pct = abs(s1-s2)/s1
    MAE = round(sm.mean_absolute_error(y_test, y_pred), 4)
    MSE = round(sm.mean_squared_error(y_test, y_pred), 4)
    MDAE = round(sm.median_absolute_error(y_test, y_pred), 4)
    EVS = round(sm.explained_variance_score(y_test, y_pred), 4)
    R2_Score = round(sm.r2_score(y_test, y_pred), 4)

    if mode == 'val':
       fw = fw_val
    elif mode == 'test':
       fw = fw_test
       
    fw.write('%15s' % method); fw.write("\t")
    fw.write('%15.4f' % s1); fw.write("\t")
    fw.write('%15.4f' % s2); fw.write("\t")
    fw.write('%15.4f' % error_pct); fw.write("\t")
    fw.write('%15.4f' % MAE); fw.write("\t")
    fw.write('%15.4f' % MSE); fw.write("\t")
    fw.write('%15.4f' % MDAE); fw.write("\t")
    fw.write('%15.4f' % EVS); fw.write("\t")
    fw.write('%15.4f' % R2_Score)
    fw.write("\n")

    print("sum of y_test = ", s1)
    print("sum of y_pred = ", s2)
    print("error pct = ", error_pct)

    print("Mean absolute error =", MAE)
    print("Mean squared error =", MSE)
    print("Median absolute error =", MDAE)
    print("Explain variance score =", EVS)
    print("R2 score =", R2_Score)
    print('--'*20)

#---------------------------------------------------------------------
import sklearn.metrics as sm
import pandas as pd
import yaml

with open('control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_path = controls['io_path']
output_path = controls['io_path']

fix_negval = True

#------------------ controls
detect_val = 'israin_val.csv' # detected rain from classification
detect_test = 'israin_test.csv'

fea_suffix = controls['fea_suffix']
time_suffix = controls['time_suffix']

#------------------ create error log file
col_names = ['Model','sum_y_val','sum_y_pred','error_pct','MAE', 'MSE', 'MDAE','EVS','R2_Score']

fw_val = open(output_path + 'rain_estimation_val_unfiltered_' + fea_suffix + time_suffix + '.log','w')
for name in col_names:
    fw_val.write('%15s' % name); fw_val.write("\t")
fw_val.write('\n')

fw_test = open(output_path + 'rain_estimation_test_unfiltered_' + fea_suffix + time_suffix + '.log','w')
for name in col_names:
    fw_test.write('%15s' % name); fw_test.write("\t")
fw_test.write('\n')

# read in data
train = pd.read_csv(input_path + 'train.csv')
val = pd.read_csv(input_path + 'val.csv')
test = pd.read_csv(input_path + 'test.csv')

rainfall_val = pd.read_csv(input_path + detect_val)
rainfall_test = pd.read_csv(input_path + detect_test)

# select features to be used for ML
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

y_train = train['rain_amount'].values
y_val = val['rain_amount'].values
y_test = test['rain_amount'].values

rainfall_val['rainfall_true'] = val['rain_amount']
rainfall_test['rainfall_true'] = test['rain_amount']

#--------------------feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_val = sc_X.transform(x_val)
x_test = sc_X.transform(x_test)

# shuffle the train data
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=100)

#--------------PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 5)
# X_train = pca.fit_transform(x_train)
# X_test = pca.transform(x_test)
# explained_variance = pca.explained_variance_ratio_
# print(sum(explained_variance))

#-----------------------------------------------------linear regression, no need for scaling
method="Linear"
print(method)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_val_res = regressor.predict(x_val)
y_test_res = regressor.predict(x_test)

y_val_res = fix_negative(fix_negval, y_val_res)
y_test_res = fix_negative(fix_negval, y_test_res)

rainfall_val['Regres_Linear'] = y_val_res
rainfall_test['Regres_Linear'] = y_test_res

cal_score(y_val, y_val_res, 'val')
cal_score(y_test, y_test_res, 'test')

#-----------------------------------------------------polynomial regression, no need for scaling
method="Polynomial"
print(method)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)
x_train_poly = poly_reg.fit_transform(x_train)

lin_reg = LinearRegression()
lin_reg.fit(x_train_poly, y_train)

x_val_poly = poly_reg.fit_transform(x_val)
y_val_res = lin_reg.predict(x_val_poly)

x_test_poly = poly_reg.fit_transform(x_test)
y_test_res = lin_reg.predict(x_test_poly)

y_val_res = fix_negative(fix_negval, y_val_res)
y_test_res = fix_negative(fix_negval, y_test_res)

rainfall_val['Regres_Poly'] = y_val_res
rainfall_test['Regres_Poly'] = y_test_res

cal_score(y_val, y_val_res, 'val')
cal_score(y_test, y_test_res, 'test')

#-----------------------------------------------------Decision Tree, no need for scaling
"""
method="DecisionTree"
print(method)

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=100)
regressor.fit(x_train,y_train)

y_val_res = regressor.predict(x_val)
y_test_res = regressor.predict(x_test)

y_val_res = fix_negative(fix_negval, y_val_res)
y_test_res = fix_negative(fix_negval, y_test_res)

rainfall_val['Regres_DT'] = y_val_res
rainfall_test['Regres_DT'] = y_test_res

cal_score(y_val, y_val_res, 'val')
cal_score(y_test, y_test_res, 'test')
"""
#-----------------------------------------------------Random Forest, no need for scaling
method="RandomForest"
print(method)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=20,random_state=0)
regressor.fit(x_train,y_train)

y_val_res = regressor.predict(x_val)
y_test_res = regressor.predict(x_test)

y_val_res = fix_negative(fix_negval, y_val_res)
y_test_res = fix_negative(fix_negval, y_test_res)

rainfall_val['Regres_RF'] = y_val_res
rainfall_test['Regres_RF'] = y_test_res

cal_score(y_val, y_val_res, 'val')
cal_score(y_test, y_test_res, 'test')

#-----------------------------------------------------SVM, require feature scaling
method="SVR"
print(method)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf') 
regressor.fit(x_train, y_train)

y_val_res = regressor.predict(x_val)
y_test_res = regressor.predict(x_test)

y_val_res = fix_negative(fix_negval, y_val_res)
y_test_res = fix_negative(fix_negval, y_test_res)

rainfall_val['Regres_SVR-rbf'] = y_val_res
rainfall_test['Regres_SVR-rbf'] = y_test_res

cal_score(y_val, y_val_res, 'val')
cal_score(y_test, y_test_res, 'test')

#----------------------------------------------------- bad performance
"""
method="SVR-poly"
print(method)

regressor=SVR(kernel='poly')
regressor.fit(x_train, y_train)

y_val_res = regressor.predict(x_val)
y_test_res = regressor.predict(x_test)

y_val_res = fix_negative(fix_negval, y_val_res)
y_test_res = fix_negative(fix_negval, y_test_res)

rainfall_val['Regres_SVR-poly'] = y_val_res
rainfall_test['Regres_SVR-poly'] = y_test_res

cal_score(y_val, y_val_res, 'val')
cal_score(y_test, y_test_res, 'test')

#-----------------------------------------------------
method="SVR-linear"
print(method)

regressor=SVR(kernel='poly')
regressor.fit(x_train, y_train)

y_val_res = regressor.predict(x_val)
y_test_res = regressor.predict(x_test)

y_val_res = fix_negative(fix_negval, y_val_res)
y_test_res = fix_negative(fix_negval, y_test_res)

rainfall_val['Regres_SVR-linear'] = y_val_res
rainfall_test['Regres_SVR-linear'] = y_test_res

cal_score(y_val, y_val_res, 'val')
cal_score(y_test, y_test_res, 'test')
"""

#-----------------------------------------------------ANN, need scaling
(dim1, dim2) = x_train.shape

method="ANN"
print(method)

# make the ANN
from keras.models import Sequential #used to initialize the ANN
from keras.layers import Dense #used to add the different layers

#initialize the ANN
regressor=Sequential()
regressor.add(Dense(units=6, activation='relu',input_dim=dim2)) #create first hidden layer

#add second hidden layer
regressor.add(Dense(units=6, activation='relu'))
#regressor.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#add the output layer
regressor.add(Dense(units=1, activation='linear'))

#compile the ANN
regressor.compile(optimizer='adam', loss='mean_squared_error')

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint(output_path+'ANN_estimator_model.best', save_best_only=True, monitor='val_loss', mode='min')

# fit the ANN to the training data
regressor.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=10, epochs=30, callbacks=[mcp_save])

regressor = keras.models.load_model(output_path+'ANN_estimator_model.best')

y_val_res = regressor.predict(x_val)
y_test_res = regressor.predict(x_test)

y_val_res = fix_negative(fix_negval, y_val_res)
y_test_res = fix_negative(fix_negval, y_test_res)

rainfall_val['Regres_ANN'] = y_val_res
rainfall_test['Regres_ANN'] = y_test_res

cal_score(y_val, y_val_res, 'val')
cal_score(y_test, y_test_res, 'test')

#------------------------------------ End
fw_val.close()
fw_test.close()

rainfall_val.to_csv(output_path + 'rainfall_val.csv', index=False)
rainfall_test.to_csv(output_path + 'rainfall_test.csv', index=False)