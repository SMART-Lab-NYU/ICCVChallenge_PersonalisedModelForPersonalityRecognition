import autokeras as ak
import tensorflow as tf
import os
import h5py
import csv
import glob
import numpy as np
import pickle
import pandas as pd
from functools import reduce
from tensorflow.keras.callbacks import CSVLogger





"""## Upload the testing metadata"""

# TESTING DATA
path = "./CSV files/final_test.csv"
print(path)

# Reading training file.
metadata_test=pd.read_csv(path)
metadata_test.drop(['Unnamed: 0'],inplace=True,axis=1)
metadata_test.sort_values(['Video'],inplace=True)
metadata_test

"""## Upload the test data"""

path_data = "./Features"
path_text_features = path_data+"/TextualFeatures/"
path_text_features_test = path_text_features+"Test/"




# Function to import data 
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
                print(file)
                fname =glob.glob(path_features+folder+'/'+file+'/output2/*.csv')
                for fn in fname:
                    print(fn)
                    df_fn = pd.read_csv(fn,index_col=None)
                    filename = fn.split('/')[7].split('.')[0]
                    print(filename)
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

# importing test data for prediction
df_sentiment_test, df_speechtrait_test, df_talkturn_test, df_time_test,df_textualall_test = import_data(path_text_features_test)




# Merging with metadata in order to maintain uniformity
def merge_with_metadata(df,metadata,HOW):
    # Merge with the metadata on the video identity
    df_=df.merge(metadata,left_on="video",right_on="Video", how=HOW)
    # Convert the ID_y to float to be able to compare it with PART.1 and PART.2
    df_['ID_y']=df_['ID_y'].astype(float)
    # Keep only the lines where PART.1 and P1 match or P2 and PART.2 match
    c1 = ((df_['PART.1']==df_["ID_y"]) & (df_["person"] == "P1"))
    c2 = ((df_['PART.2']==df_["ID_y"]) & (df_["person"] == "P2"))
    c = c1 | c2
    df_ = df_[c]
    df_.drop_duplicates(subset =["video","context","person","minute"],inplace = True)
    # Remove unnecessary columns
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
    

# Separating and preparing data for prediction
def prepare_data_for_ML(df,dataset):
    if dataset=='test':
        X = df.to_numpy()
        X = np.asarray(X).astype(np.float32)
        return X
    




df_alltextual_test_merged = merge_with_metadata(df_textualall_test,metadata_test,"left")

# Looping through each directory and loading the files using above defined function
for iter_feature in ['alltextual']:
    for iter_set in ['test']:
        print(f"Feature: {iter_feature}, Set: {iter_set}")
        exec(f"df_{iter_feature}_{iter_set}_male,df_{iter_feature}_{iter_set}_female,df_{iter_feature}_{iter_set}_male_meta,df_{iter_feature}_{iter_set}_female_meta=create_profile_dataset(df_{iter_feature}_{iter_set}_merged,'GENDER')")
        for iter_gender in ['male','female']:
            exec(f"{iter_gender}_{iter_set}_{iter_feature}_X=prepare_data_for_ML(df_{iter_feature}_{iter_set}_{iter_gender},iter_set)")




model_dir='./Textual_pretrained_models'
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
    



# Function that returns predicted results
def predict(X_test,model_filename,csv_logger, model_reg):
    fit_results=None
    predicted_y_test = model_reg.predict(X_test)
    print(model_reg.summary)
    del model_reg
    return predicted_y_test




i = 0
# Looping through male and female alltextual models to predict and append results. 
for iter_gender in ['male','female']:
    for iter_feature in ['alltextual']:#
        for iter_pers_label in ['o','c','e','a','n']:
            print(f"Personality Label: {iter_pers_label}, Gender: {iter_gender}, Feature: {iter_feature}")
            logfilename = results_dir+f"/fit_results_{iter_feature}_{iter_pers_label}_{iter_gender}.csv"
            model_filename = f"./Textual_pretrained_models/{iter_feature}_{iter_gender}_{iter_pers_label}"
            model = tf.keras.models.load_model(model_filename)
            csv_logger = CSVLogger(logfilename, append=True, separator=';')
            exec(f"predicted_test_{iter_feature}_{iter_pers_label}_{iter_gender}=predict({iter_gender}_test_{iter_feature}_X,model_filename,csv_logger, model)")
            i = i+1
            if (iter_pers_label=='o'):
                exec(f"df_predicted_test_{iter_feature}_{iter_gender} = pd.DataFrame(data=predicted_test_{iter_feature}_{iter_pers_label}_{iter_gender},columns=['o'])") 
            else:
                exec(f"df_predicted_test_{iter_feature}_{iter_gender}[iter_pers_label]=predicted_test_{iter_feature}_{iter_pers_label}_{iter_gender}")
            
        exec(f"df_predicted_test_{iter_feature}_{iter_gender}.to_csv('{results_dir}/{iter_feature}_{iter_gender}_test.csv',index=False)")

   


# Saving the metadata for future use
df_alltextual_test_male_meta.to_csv('./Results/test_male_meta.csv')
df_alltextual_test_female_meta.to_csv('./Results/test_female_meta.csv')



# Concatenating both the results files
model_dir='./Textual_pretrained_models'
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














