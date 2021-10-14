# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.run import Run
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import xgboost as xgb



# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"


def clean_data(df):
    
    # replace blanks with np.nan
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    # convert to float64
    df['TotalCharges'] = df['TotalCharges'].astype('float64')
    
    # Replace binary values
    df['gender'] = df['gender'].apply(lambda s: 1 if s == "Female" else 0)
    df['Partner'] = df['Partner'].apply(lambda s: 1 if s == "Yes" else 0)
    df['Dependents'] = df['Dependents'].apply(lambda s: 1 if s == "Yes" else 0)
    df['PhoneService'] = df['PhoneService'].apply(lambda s: 1 if s == "Yes" else 0)
    df['PaperlessBilling'] = df['PaperlessBilling'].apply(lambda s: 1 if s == "Yes" else 0)
    df['Churn'] = df['Churn'].apply(lambda s: 1 if s == "Yes" else 0)

    MultipleLines = pd.get_dummies(df['MultipleLines'], prefix="MultipleLines")
    df.drop("MultipleLines", inplace=True, axis=1)
    df = df.join(MultipleLines)
    InternetService = pd.get_dummies(df['InternetService'], prefix="InternetService")
    df.drop("InternetService", inplace=True, axis=1)
    df = df.join(InternetService)
    OnlineSecurity = pd.get_dummies(df['OnlineSecurity'], prefix="OnlineSecurity")
    df.drop("OnlineSecurity", inplace=True, axis=1)
    df = df.join(OnlineSecurity)
    OnlineBackup = pd.get_dummies(df['OnlineBackup'], prefix="OnlineBackup")
    df.drop("OnlineBackup", inplace=True, axis=1)
    df = df.join(OnlineBackup)
    DeviceProtection = pd.get_dummies(df['DeviceProtection'], prefix="DeviceProtection")
    df.drop("DeviceProtection", inplace=True, axis=1)
    df = df.join(DeviceProtection)
    TechSupport = pd.get_dummies(df['TechSupport'], prefix="TechSupport")
    df.drop("TechSupport", inplace=True, axis=1)
    df = df.join(TechSupport)
    StreamingTV = pd.get_dummies(df['StreamingTV'], prefix="StreamingTV")
    df.drop("StreamingTV", inplace=True, axis=1)
    df = df.join(StreamingTV)
    StreamingMovies = pd.get_dummies(df['StreamingMovies'], prefix="StreamingMovies")
    df.drop("StreamingMovies", inplace=True, axis=1)
    df = df.join(StreamingMovies)
    Contract = pd.get_dummies(df['Contract'], prefix="Contract")
    df.drop("Contract", inplace=True, axis=1)
    df = df.join(Contract)
    PaymentMethod = pd.get_dummies(df['PaymentMethod'], prefix="PaymentMethod")
    df.drop("PaymentMethod", inplace=True, axis=1)
    df = df.join(PaymentMethod)
    y_df = df.pop('Churn')
    # x_df = df.drop("Churn", inplace=True, axis=1)

    return df, y_df


ds= TabularDatasetFactory.from_delimited_files("https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv")

x, y = clean_data(ds)

# TODO: Split data into train, valid and, test sets.

# Retrieve datasets by name | Create train/val/test
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, stratify=y, random_state=40)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.2, stratify=y_val, random_state=40)

run = Run.get_context()

def main():
    # Add arguments to script 
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--input_data_name", type=str, help="Environment name for cleaned loan dataset")
    
    # XGBClassifier args
    parser.add_argument("--eta", type=float, help="Learning rate for model")
    parser.add_argument("--max_depth", type=int, help="Depth for trees")
    parser.add_argument("--min_child_weight", type=int, help="Min child weight for tree")
    parser.add_argument("--subsample", type=float, help="Subsample of training set used for each iteration")
    parser.add_argument("--colsample_bytree", type=float, help="Subsample of columns to use for each iteration")
    parser.add_argument("--early_stopping_rounds", type=int, help="Model will stop iterating if no improvement after set number of rounds")
    parser.add_argument("--eval_metric", type=str, default="auc", help="Metric for evaluation")
    parser.add_argument("--scale_pos_weight", type=float, help="Control balance of positive and negative weights")
    parser.add_argument("--max_delta_step", type=int, help="Conservativeness of update step")
    args = parser.parse_args()

    # create DMatrix objects 
 
    dtrain = xgb.DMatrix(x_train.values, label=y_train.values)
    dval = xgb.DMatrix(x_val.values, label=y_val.values)
    dtest = xgb.DMatrix(x_test.values, label=y_test.values)

    # Log model parameters
    run.log("max_depth", int(args.max_depth))
    run.log("min_child_weight", int(args.min_child_weight))
    run.log("subsample", np.float(args.subsample))
    run.log("colsample_bytree", np.float(args.colsample_bytree))
    run.log("scale_pos_weight", np.float(args.scale_pos_weight))
    run.log("max_delta_step", int(args.max_delta_step))
    run.log("eta", np.float(args.eta))

    params = {
        "objective": "binary:logistic",
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": args.eval_metric,
        "scale_pos_weight": args.scale_pos_weight,
        "max_delta_step": args.max_delta_step,
    }

    # set to an arbitrarily large number
    num_boost_rounds = 999

    # train to find best number of boost rounds
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_rounds,
        evals=[(dval, "Val")],
        early_stopping_rounds=args.early_stopping_rounds
    )
    # get optimal number of boost rounds
    run.log("num_boost_rounds", model.best_iteration+1)

    # Make prediction on Val dataset & log AUC
    y_pred = [1 if x >= 0.5 else 0 for x in model.predict(dtest, ntree_limit=model.best_ntree_limit)]
    auc_score = roc_auc_score(y_test, y_pred, average="weighted")
    run.log("auc", np.float(auc_score))

    print("Classification Report: \n", classification_report(y_test, y_pred))

    # Dump model artifact 
    os.makedirs('outputs/hyperdrive', exist_ok=True)
    joblib.dump(model, "outputs/hyperdrive/model.joblib")

if __name__ == "__main__":
    main()
