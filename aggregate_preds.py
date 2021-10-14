from typing import final
import pandas as pd

path_to_fb = "./Results/fb_pred.csv"

path_to_textual = "./Results/alltextual_results.csv"

df_fb = pd.read_csv(path_to_fb)
df_textual = pd.read_csv(path_to_textual)

df_fb = df_fb.reset_index(drop= True)

df_fb = df_fb.drop(["Unnamed: 0","minute"], axis = 1)
mean_df_fb = df_fb.groupby(['ID_y'], as_index=False).mean()
mean_df_fb.columns = [0, 1, 2, 3, 4, 5]

print("textual")
df_textual.drop([df_textual.columns[0],df_textual.columns[2]], axis = 1, inplace = True)
print("After dropping:")
df_textual.columns = [0, 1, 2, 3, 4, 5]
print(df_textual.head())
mean_df_text = df_textual.groupby(df_textual.columns[0], as_index=False).mean()

final_pred = mean_df_text.append(mean_df_fb)
final_pred = final_pred.groupby(final_pred.columns[0], as_index=False).mean()
final_pred.columns = ["Participant ID",	"OPENMINDEDNESS_Z",	"CONSCIENTIOUSNESS_Z",	"EXTRAVERSION_Z",	"AGREEABLENESS_Z",	"NEGATIVEEMOTIONALITY_Z"]
print(final_pred.head(20))
final_pred.to_csv("./final_predictions.csv")
