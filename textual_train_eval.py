import autokeras as ak
import tensorflow as tf
import os
import h5py
import numpy as np
import pickle
import pandas as pd
from functools import reduce




"""## Upload the validation and training metadata"""

# VALIDATION DATA
path = "./CSV files/Final_val.csv"
print(path)
metadata_valid = pd.read_csv(path)
metadata_valid.drop(['Unnamed: 0','TIMESTAMP','COUNTRY','EDUCATION','LANGUAGE'],inplace=True,axis=1)
metadata_valid.sort_values(['Video'],inplace=True)

metadata_valid.head()


# TRAINING DATA
path = "./CSV files/Final-3.csv"
print(path)

# Reading training file.
metadata_train=pd.read_csv(path)
metadata_train.drop(['Unnamed: 0','LANGUAGE'],inplace=True,axis=1)
metadata_train.sort_values(['Video'],inplace=True)
metadata_train

# TESTING DATA
path = "./CSV files/final_test.csv"
print(path)

# Reading training file.
metadata_test=pd.read_csv(path)
metadata_test.drop(['Unnamed: 0'],inplace=True,axis=1)
metadata_test.sort_values(['Video'],inplace=True)
metadata_test

"""## Upload the training, validation and test data"""
path_data = "./Features"
path_text_features = path_data+"/TextualFeatures/"

path_text_features_train = path_text_features+"Train/"
path_text_features_valid = path_text_features+"Validation/"
path_text_features_test = path_text_features+"Test/"

import glob





def import_data(path_features):
    
    listing=os.listdir(path_features)

    df_sentiment = pd.DataFrame(columns = ["video","context","person","minute","SentiMin", "SentiMax", "SentiAvg", "SentiStdev", "Senti0-0.20", "Senti0.21-0.40", "Senti0.41-0.60", "Senti0.61-0.80", "Senti0.81-1.00", "SentiTotal"])
    df_speechtrait = pd.DataFrame(columns =["video","context","person","minute","TraitMin","TraitMax","TraitAvg","TraitStdev","TraitTotal"])
    df_talkturn = pd.DataFrame(columns =["video","context","person","minute","TurnPercent","AvgWrdPerTurn","LongestTurn","TotalWrds","STDWrdPerTurn"])
    df_time = pd.DataFrame(columns=["video","context","person","minute","TimeMin","TimeMax","TimeAvg","TimeStdev","TimeTotal"])

    i = 0
    for folder in listing:
        if folder == ".DS_Store":
            continue
        else:
            print(folder)
            file_listing = os.listdir(path_features+folder+"/")
            for file in file_listing:
                fname =glob.glob(path_features+folder+'/'+file+'/output2/*.csv')
                for fn in fname:
                    df_fn = pd.read_csv(fn,index_col=None)
                    filename = fn.split('/')[7].split('.')[0]
                    feature_name = filename.split('_')[4]
                    if feature_name == "RawSenti":
                        df_sentiment.at[i,"video"]=filename.split('_')[0]
                        df_sentiment.at[i,"context"]=filename.split('_')[1]
                        df_sentiment.at[i,"person"]=filename.split('_')[2]
                        df_sentiment.at[i,"minute"]=filename.split('_')[3]
                        df_sentiment.at[i,4:]=df_fn.values
                    elif feature_name == "SpeechTrait":
                        df_speechtrait.at[i,"video"]=filename.split('_')[0]
                        df_speechtrait.at[i,"context"]=filename.split('_')[1]
                        df_speechtrait.at[i,"person"]=filename.split('_')[2]
                        df_speechtrait.at[i,"minute"]=filename.split('_')[3]
                        df_speechtrait.at[i,4:]=df_fn.values
                    elif feature_name == "TalkTurn":
                        df_talkturn.at[i,"video"]=filename.split('_')[0]
                        df_talkturn.at[i,"context"]=filename.split('_')[1]
                        df_talkturn.at[i,"person"]=filename.split('_')[2]
                        df_talkturn.at[i,"minute"]=filename.split('_')[3]
                        df_talkturn.at[i,4:]=df_fn.values
                    elif feature_name == "Time":
                        df_time.at[i,"video"]=filename.split('_')[0]
                        df_time.at[i,"context"]=filename.split('_')[1]
                        df_time.at[i,"person"]=filename.split('_')[2]
                        df_time.at[i,"minute"]=filename.split('_')[3]
                        df_time.at[i,4:]=df_fn.values
                    i = i+1
    df_sentiment['video']=df_sentiment['video'].astype(float)
    df_speechtrait['video']=df_speechtrait['video'].astype(float)
    df_talkturn['video']=df_talkturn['video'].astype(float)
    df_time['video']=df_time['video'].astype(float)
    df_sentiment.sort_values(['video', 'context','person','minute'],inplace=True)
    df_speechtrait.sort_values(['video', 'context','person','minute'],inplace=True)
    df_talkturn.sort_values(['video', 'context','person','minute'],inplace=True)
    df_time.sort_values(['video', 'context','person','minute'],inplace=True)
    df_textualall=pd.concat([df_sentiment.reset_index(drop=True),df_speechtrait.reset_index(drop=True),df_talkturn.reset_index(drop=True),df_time.reset_index(drop=True)], axis=1)
    df_textualall = df_textualall.loc[:,~df_textualall.columns.duplicated()]
    df_sentiment.dropna(inplace=True)
    df_speechtrait.dropna(inplace=True)
    df_talkturn.dropna(inplace=True)
    df_time.dropna(inplace=True)
    df_textualall.dropna(inplace=True)
    df_sentiment[['Senti0-0.20', 'Senti0.21-0.40','Senti0.41-0.60', 'Senti0.61-0.80', 'Senti0.81-1.00']]= df_sentiment[['Senti0-0.20', 'Senti0.21-0.40','Senti0.41-0.60', 'Senti0.61-0.80', 'Senti0.81-1.00']].div(df_sentiment[['Senti0-0.20', 'Senti0.21-0.40','Senti0.41-0.60', 'Senti0.61-0.80', 'Senti0.81-1.00']].sum(axis=1), axis=0)
    return df_sentiment, df_speechtrait, df_talkturn, df_time,df_textualall

df_sentiment_train, df_speechtrait_train, df_talkturn_train, df_time_train,df_textualall_train = import_data(path_text_features_train)

df_sentiment_valid, df_speechtrait_valid, df_talkturn_valid, df_time_valid,df_textualall_valid = import_data(path_text_features_valid)

df_sentiment_test, df_speechtrait_test, df_talkturn_test, df_time_test,df_textualall_test = import_data(path_text_features_test)




# Merging with metadata to maintain uniformity
def merge_with_metadata(df,metadata,HOW):
    #merge with the metadata on the video identity
    df_=df.merge(metadata,left_on="video",right_on="Video", how=HOW)
    #convert the ID_y to float to be able to compare it with PART.1 and PART.2
    df_['ID_y']=df_['ID_y'].astype(float)
    #keep only the lines where PART.1 and P1 match or P2 and PART.2 match
    c1 = ((df_['PART.1']==df_["ID_y"]) & (df_["person"] == "P1"))
    c2 = ((df_['PART.2']==df_["ID_y"]) & (df_["person"] == "P2"))
    c = c1 | c2
    df_ = df_[c]
    df_.drop_duplicates(subset =["video","context","person","minute"],inplace = True)
    #Remove unnecessary columns
    df_.drop(['SESSION1', 'SESSION2','SESSION3','SESSION4','Video','video','PART.1','PART.2','person'], axis = 1,inplace=True)
    return df_

# Create different datasets according to the profile
def create_profile_dataset(df,column):
    if (column=='GENDER'):
        df_m=df[df['GENDER']=='M']
        df_f=df[df['GENDER']=='F']
        df_m_meta = df_m[['ID_y','minute']].copy()
        df_f_meta = df_f[['ID_y','minute']].copy()
        df_m.drop(['GENDER','context','AGE','ID_y','minute'], axis = 1,inplace=True)
        df_f.drop(['GENDER','context','AGE','ID_y','minute'], axis = 1,inplace=True)
        return df_m,df_f,df_m_meta,df_f_meta
    

import csv
def prepare_data_for_ML(df,dataset):
    if dataset=='test':
        X = df.to_numpy()
        X = np.asarray(X).astype(np.float32)
        return X
    else:
        y_o = df.iloc[:,-5].to_numpy()
        y_c = df.iloc[:,-4].to_numpy()
        y_e = df.iloc[:,-3].to_numpy()
        y_a = df.iloc[:,-2].to_numpy()
        y_n = df.iloc[:,-1].to_numpy()
        ocean = df.iloc[:,-5:]
        X   = df.iloc[:,:-5].to_numpy()
        X = np.asarray(X).astype(np.float32)
        o_set = tf.data.Dataset.from_tensor_slices((X, y_o))
        c_set = tf.data.Dataset.from_tensor_slices((X, y_c))
        e_set = tf.data.Dataset.from_tensor_slices((X, y_e))
        a_set = tf.data.Dataset.from_tensor_slices((X, y_a))
        n_set = tf.data.Dataset.from_tensor_slices((X, y_n))
        return o_set,c_set,e_set,a_set,n_set,ocean



df_alltextual_valid_merged = merge_with_metadata(df_textualall_valid,metadata_valid,"left")
df_alltextual_train_merged = merge_with_metadata(df_textualall_train,metadata_train,"left")
df_alltextual_test_merged = merge_with_metadata(df_textualall_test,metadata_test,"left")

for iter_feature in ['alltextual']:
    for iter_set in ['valid','train','test']:
        print(f"Feature: {iter_feature}, Set: {iter_set}")
        exec(f"df_{iter_feature}_{iter_set}_male,df_{iter_feature}_{iter_set}_female,df_{iter_feature}_{iter_set}_male_meta,df_{iter_feature}_{iter_set}_female_meta=create_profile_dataset(df_{iter_feature}_{iter_set}_merged,'GENDER')")
        for iter_gender in ['male','female']:
            if iter_set=='test':
                exec(f"{iter_gender}_{iter_set}_{iter_feature}_X=prepare_data_for_ML(df_{iter_feature}_{iter_set}_{iter_gender},iter_set)")
            else:
                exec(f"{iter_gender}_{iter_set}_{iter_feature}_o,{iter_gender}_{iter_set}_{iter_feature}_c,{iter_gender}_{iter_set}_{iter_feature}_e,{iter_gender}_{iter_set}_{iter_feature}_a,{iter_gender}_{iter_set}_{iter_feature}_n,ocean_{iter_gender}_{iter_set}=prepare_data_for_ML(df_{iter_feature}_{iter_set}_{iter_gender},iter_set)")
 




            
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

# Creating/Defining paths to store model and results
model_dir='./Models'
results_dir = "./Results"
df_validation_loss =  pd.DataFrame(columns = ["Pers_Label","Feature","Profile_Split","val_loss"])

try:
    os.mkdir(results_dir)
except OSError:
    print ("Creation of the directory %s failed" % results_dir)
else:
    print ("Successfully created the directory %s " % results_dir)
    
try:
    os.mkdir(model_dir)
except OSError:
    print ("Creation of the directory %s failed" % model_dir)
else:
    print ("Successfully created the directory %s " % model_dir)    
    



"""## Training, validation and prediction"""
# Training on the train set, evaluating on the validation set, and predicting on the test set and validation set
nb_trial = 100
nb_epochs = 1000

def fit_and_evaluate(tf_train_set,tf_val_set,X_test,TRIALS,EPOCHS,model_filename,csv_logger):
    fit_results=None
    model_reg = None
    model_reg = ak.StructuredDataRegressor(max_trials=TRIALS, overwrite=True)
    fit_results = model_reg.fit(tf_train_set, epochs=EPOCHS, validation_split=0.15,callbacks=[csv_logger])#validation_data=tf_val_set
    validation_loss = model_reg.evaluate(tf_val_set)
    predicted_y_val = model_reg.predict(tf_val_set)
    predicted_y_test = model_reg.predict(X_test)
    model = model_reg.export_model()
    print(type(model)) 
    try:
        model.save(model_filename, save_format="tf")
    except Exception:
        model.save(model_filename+".h5")
    del model_reg
    del model
    return fit_results,validation_loss,predicted_y_val,predicted_y_test




i = 0

for iter_gender in ['male','female']:
    for iter_feature in ['alltextual']:
        for iter_pers_label in ['o','c','e','a','n']:
            print(f"Personality Label: {iter_pers_label}, Gender: {iter_gender}, Feature: {iter_feature}")
            logfilename = results_dir+f"/fit_results_{iter_feature}_{iter_pers_label}_{iter_gender}.csv"
            model_filename = f"./Models/{iter_feature}_{iter_gender}_{iter_pers_label}"
            csv_logger = CSVLogger(logfilename, append=True, separator=';')
            exec(f"fit_results_{iter_feature}_{iter_pers_label}_{iter_gender},val_loss_{iter_feature}_{iter_pers_label}_{iter_gender},predicted_val_{iter_feature}_{iter_pers_label}_{iter_gender},predicted_test_{iter_feature}_{iter_pers_label}_{iter_gender}=fit_and_evaluate({iter_gender}_train_{iter_feature}_{iter_pers_label},{iter_gender}_valid_{iter_feature}_{iter_pers_label},{iter_gender}_test_{iter_feature}_X,nb_trial,nb_epochs,model_filename,csv_logger)")
            exec(f"df_validation_loss.at[i,'val_loss']=val_loss_{iter_feature}_{iter_pers_label}_{iter_gender}")
            exec(f"df_validation_loss.at[i,'Pers_Label']=iter_pers_label")
            exec(f"df_validation_loss.at[i,'Feature']=iter_feature")
            exec(f"df_validation_loss.at[i,'Profile_Split']=iter_gender")
            i = i+1
            if (iter_pers_label=='o'):
                exec(f"df_predicted_val_{iter_feature}_{iter_gender} = pd.DataFrame(data=predicted_val_{iter_feature}_{iter_pers_label}_{iter_gender},columns=['o'])") 
                exec(f"df_predicted_test_{iter_feature}_{iter_gender} = pd.DataFrame(data=predicted_test_{iter_feature}_{iter_pers_label}_{iter_gender},columns=['o'])") 
            else:
                exec(f"df_predicted_val_{iter_feature}_{iter_gender}[iter_pers_label]=predicted_val_{iter_feature}_{iter_pers_label}_{iter_gender}")
                exec(f"df_predicted_test_{iter_feature}_{iter_gender}[iter_pers_label]=predicted_test_{iter_feature}_{iter_pers_label}_{iter_gender}")
            
        exec(f"df_predicted_val_{iter_feature}_{iter_gender}.to_csv('{results_dir}/{iter_feature}_{iter_gender}_val.csv',index=False)")
        exec(f"df_predicted_test_{iter_feature}_{iter_gender}.to_csv('{results_dir}/{iter_feature}_{iter_gender}_test.csv',index=False)")

   



df_alltextual_test_male_meta.to_csv('./Results/test_male_meta.csv')
df_alltextual_test_female_meta.to_csv('./Results/test_female_meta.csv')




model_dir='./Models'
results_dir = "./Results"


df_alltextual_test_male_meta=pd.read_csv('./Results/test_male_meta.csv')
df_alltextual_test_female_meta=pd.read_csv('./Results/test_female_meta.csv')

for iter_gender in ['male','female']:
    for iter_feature in ['alltextual']:
        print(f"Gender: {iter_gender}, Feature: {iter_feature}")
        resfilename = results_dir+f"/{iter_feature}_{iter_gender}_test.csv"
        exec(f"df_results_{iter_feature}_{iter_gender} = pd.read_csv(resfilename,sep=',')")

alltextual_female = pd.concat([df_alltextual_test_female_meta.reset_index(drop=True),df_results_alltextual_female.reset_index(drop=True)], axis=1)

alltextual_male = pd.concat([df_alltextual_test_male_meta.reset_index(drop=True),df_results_alltextual_male.reset_index(drop=True)], axis=1)

alltextual_male.drop(["Unnamed: 0"],axis = 1,inplace=True)

alltextual_female.drop(["Unnamed: 0"],axis = 1,inplace=True)

alltextual = alltextual_male.append(alltextual_female)

alltextual.to_csv("./Results/alltextual_results.csv")

alltextual






