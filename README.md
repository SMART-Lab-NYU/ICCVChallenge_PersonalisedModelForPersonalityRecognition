# ICCV Challenge Personalised Model For Personality Recognition
This is the repository to the winning solution of the ICCV'21 DYAD Challenge for Personalised Model For Personality Recognition. 


---
## Set-up instructions

### Data: 
The dataset used to train the models is the UDIVA_v0.5 dataset

### Code
The repository contains a requirements.txt file that mentions all the necessary libraries on Python 3.8 that are required to replicate the results as well as conduct future experiments. 

There are two bash files. 

a. **run_train_eval.sh** - Running this file will train new models for both face and body features as well as the textual features.

b. **run_eval.sh** - Running this file will only predict on the test data using pre-trained models. 

## Results
The results are aggregated automatically using the **aggregate_preds.py** file and stores in the main directory as **final_predictions.csv**

## Fact Sheet

For more information about the dataset used and the preprocessing, the training process and the results, please check our [fact-sheet](https://drive.google.com/file/d/1iXn-wJ8OYlD9i_5yLsbOuzleQnl9ZOyO/view?usp=sharing). 

