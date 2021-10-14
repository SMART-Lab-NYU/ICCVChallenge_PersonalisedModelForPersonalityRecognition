#!/bin/bash
# This file is to predict on previously trained models for face and body. The textual features don't have pretrained models so they will have to be trained regardless.

# Run next command if you want to predict on previously trained face body models
exec python3 ./fb_predict_pretrained.py
# Run this predict textual features
exec python3 ./textual_eval.py
# Run this to get final predictions
exec python3 ./aggregate_preds.py

