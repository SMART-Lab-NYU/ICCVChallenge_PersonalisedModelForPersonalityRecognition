import os
import numpy as np
import pandas as pd
import autokeras as ak
import tensorflow as tf

# Reading files. 
df_train=pd.read_csv('./CSV files/train_data.csv')
df_validation=pd.read_csv('./CSV files/validation_data.csv')
df_test=pd.read_csv('./CSV files/test_data.csv')

cols = list(df_validation.columns)


# Split it into male and female dataframes

M_train = df_train[df_train['gender'] == 'M']
F_train = df_train[df_train['gender'] == 'F']

M_val = df_validation[df_validation['gender'] == 'M']
F_val = df_validation[df_validation['gender'] == 'F']

M_test = df_test[df_test['gender'] == 'M']
F_test = df_test[df_test['gender'] == 'F']

# Labels def
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


M_test_np = np.array(M_test)
F_test_np = np.array(F_test)
M_val_np = np.array(M_val)
F_val_np = np.array(F_val)

OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']


"""# Making predictions based on female models for OCEAN individually



"""

# Change this according to the path of male and female
female_path = './FaceBody_pretrained_models/Female/'
male_path = './FaceBody_pretrained_models/Male/'

# change this to whatever you have named the folders where each model is inside male and female
OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']
# Make sure they have similar names
for mod in OCEAN_models:
    model = tf.keras.models.load_model(female_path+mod)
    model.summary
    prediction = model.predict(F_test_np)
    print(len(prediction))
    prediction = np.array(prediction)
    F_test_data = pd.concat([F_test_data, pd.DataFrame(prediction)], axis=1)
# Here F_test_data is only the ID and minute columns of the test data you're passing in, which we are merging the 
# predictions with. 
# Make sure you change it to whatever your dataframe is.

"""# Making predictions based on female models for OCEAN individually"""

for mod in OCEAN_models:
    model = tf.keras.models.load_model(male_path+mod)
    prediction = model.predict(M_test_np)
    print(len(prediction))
    prediction = np.array(prediction)
    M_test_data = pd.concat([M_test_data, pd.DataFrame(prediction)], axis=1)

total_test_data = M_test_data.append(F_test_data)
total_test_data=total_test_data.sort_values(by=['ID_y'])
total_test_data.to_csv("./Results/fb_pred.csv")
print("Prediction over")
