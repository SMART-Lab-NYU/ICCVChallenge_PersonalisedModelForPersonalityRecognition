#!/bin/bash
# Run face body training model
exec python3 ./fb_train.py
# Run this if you want to predict on new models
exec python3 ./fb_predict.py
# Run this to trian and predict textual features
exec python3 ./textual_train_eval.py
# Run this to get final predictions
exec python3 ./aggregate_preds.py

