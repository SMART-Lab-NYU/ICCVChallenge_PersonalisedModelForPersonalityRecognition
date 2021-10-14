import os
import h5py
import numpy as np
import pickle
import autokeras as ak
import tensorflow as tf
import pandas as pd



# Reading files. 
df_train=pd.read_csv('./CSV files/train_data.csv')
df_validation=pd.read_csv('./CSV files/validation_data.csv')
df_test=pd.read_csv('./CSV files/test_data.csv')

cols = list(df_validation.columns)
df_test.head()

"""# Data Handling"""

# Split it into male and female dataframes

M_train = df_train[df_train['gender'] == 'M']
F_train = df_train[df_train['gender'] == 'F']

M_val = df_validation[df_validation['gender'] == 'M']
F_val = df_validation[df_validation['gender'] == 'F']

M_test = df_test[df_test['gender'] == 'M']
F_test = df_test[df_test['gender'] == 'F']

print(df_test.head())

# Labels definition
labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z']
# male train OCEAN 
M_train_O = M_train.pop(labels[0]).to_numpy()
M_train_C = M_train.pop(labels[1]).to_numpy()
M_train_E = M_train.pop(labels[2]).to_numpy()
M_train_A = M_train.pop(labels[3]).to_numpy()
M_train_N= M_train.pop(labels[4]).to_numpy()

# female train OCEAN
F_train_O = F_train.pop(labels[0]).to_numpy()
F_train_C = F_train.pop(labels[1]).to_numpy()
F_train_E = F_train.pop(labels[2]).to_numpy()
F_train_A = F_train.pop(labels[3]).to_numpy()
F_train_N= F_train.pop(labels[4]).to_numpy()

# male validation OCEAN

M_val_O = M_val.pop(labels[0]).to_numpy()
M_val_C = M_val.pop(labels[1]).to_numpy()
M_val_E = M_val.pop(labels[2]).to_numpy()
M_val_A = M_val.pop(labels[3]).to_numpy()
M_val_N = M_val.pop(labels[4]).to_numpy()

# female validation OCEAN

F_val_O = F_val.pop(labels[0]).to_numpy()
F_val_C = F_val.pop(labels[1]).to_numpy()
F_val_E = F_val.pop(labels[2]).to_numpy()
F_val_A = F_val.pop(labels[3]).to_numpy()
F_val_N = F_val.pop(labels[4]).to_numpy()

# Modifying and arranging data
drop_cols = ['Unnamed: 0', 'Unnamed: 0.1']
drop_cols_test = ['Unnamed: 0']
data_cols = [ 'ID_y', 'minute']

M_train_data = M_train[data_cols]
F_train_data = F_train[data_cols]
M_val_data = M_val[data_cols]
F_val_data = F_val[data_cols]
M_test_data = M_test[data_cols]
F_test_data = F_test[data_cols]

data_cols = [ 'Video', 'ID_y', 'minute', 'session',
       'gender']

M_train = M_train.drop(data_cols+drop_cols, axis = 1)
F_train = F_train.drop(data_cols+drop_cols, axis = 1)
M_val = M_val.drop(data_cols+drop_cols, axis = 1)
F_val = F_val.drop(data_cols+drop_cols, axis = 1)
M_test = M_test.drop(data_cols+drop_cols_test, axis = 1)
F_test = F_test.drop(data_cols+drop_cols_test, axis = 1)

M_val.reset_index(drop=True, inplace=True)
F_val.reset_index(drop=True, inplace=True)
M_val_data.reset_index(drop=True, inplace=True)
F_val_data.reset_index(drop=True, inplace=True)
F_test_data.reset_index(drop = True, inplace = True)
M_test_data.reset_index(drop = True, inplace = True)

print(F_test_data.head())

M_test_np = np.array(M_test)
F_test_np = np.array(F_test)
M_val_np = np.array(M_val)
F_val_np = np.array(F_val)

OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']

M_train_np = np.array(M_train)
F_train_np = np.array(F_train)
M_val_np = np.array(M_val)
F_val_np = np.array(F_val)

# COnverting to tensorflow datasets
maleO_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_O))
maleC_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_C))
maleE_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_E))
maleA_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_A))
maleN_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_N))

femaleO_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_O))
femaleC_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_C))
femaleE_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_E))
femaleA_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_A))
femaleN_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_N))

maleO_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_O))
maleC_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_C))
maleE_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_E))
maleA_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_A))
maleN_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_N))

femaleO_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_O))
femaleC_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_C))
femaleE_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_E))
femaleA_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_A))
femaleN_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_N))

print(F_test_np.shape)

male_train_set = [maleO_train_set, 
                  maleC_train_set, 
                  maleE_train_set, 
                  maleA_train_set, 
                  maleN_train_set]
male_validation_set = [maleO_validation_set, 
                       maleC_validation_set, 
                       maleE_validation_set, 
                       maleA_validation_set, 
                       maleN_validation_set]

female_train_set = [femaleO_train_set, 
                    femaleC_train_set, 
                    femaleE_train_set,
                    femaleA_train_set,
                    femaleN_train_set]
female_validation_set = [femaleO_validation_set,
                         femaleC_validation_set,
                         femaleE_validation_set,
                         femaleA_validation_set,
                         femaleN_validation_set]

model_names = ['Model_O','Model_C','Model_E','Model_A','Model_N']

"""# Train
Currently:  
Validation split: 0.25  
Epochs: 1000  
Trials: 100  
## Male models
"""
print("MALE TRAINING BEGINS")

for i in range(5):
  train = male_train_set[i]
  val = male_validation_set[i]
  print(model_names[i])
  # Define a regressor
  total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = 'male'+model_names[i], directory = './Dump Data')
  # Feed the tensorflow Dataset to the regressor.
  total_reg.fit(train, epochs=1000, validation_split=0.25)
  # Convert to model   
  total_model = total_reg.export_model()
  # Evaluate on validation set
  evaluation = total_reg.evaluate(val)
  # Write loss and error to a file
  with open('./Dump Data/male_eval_val.txt', 'a') as f:
      f.write('male'+model_names[i]+' -> ')
      f.write(str(evaluation))
      f.write('\n')
  # Save current model
  total_model.save('./FaceBody_models/'+model_names[i])

"""## Female models"""
print("FEMALE TRAINING BEGINS")

for i in range(5):
  train = female_train_set[i]
  val = female_validation_set[i]
  # Define a regressor
  total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = 'female'+model_names[i], directory = './Dump Data')
  # Feed the tensorflow Dataset to the regressor.
  total_reg.fit(train, epochs=1000, validation_split=0.25)
  # Convert to model   
  total_model = total_reg.export_model()
  # Evaluate on validation set
  evaluation = total_reg.evaluate(val)
  # Write loss and error to a file
  with open('./Dump Data/female_eval_val.txt', 'a') as f:
      f.write('female'+model_names[i]+' -> ')
      f.write(str(evaluation))
      f.write('\n')
  # Save current model  
  total_model.save('./FaceBody_models/'+model_names[i])
print("TRAINING DONE")
