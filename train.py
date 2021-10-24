.reshape(1, -1)# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(df):
    df.drop('customerID', axis=1, inplace=True)
    # Replace binary values
    df["gender"] = df["gender"].apply(lambda s: 1 if s == "Female" else 0)
    df["Partner"] = df["Partner"].apply(lambda s: 1 if s == "Yes" else 0)
    df["Dependents"] = df["Dependents"].apply(lambda s: 1 if s == "Yes" else 0)
    df["PhoneService"] = df["PhoneService"].apply(lambda s: 1 if s == "Yes" else 0)
    df["PaperlessBilling"] = df["PaperlessBilling"].apply(
        lambda s: 1 if s == "Yes" else 0
    )
    df["Churn"] = df["Churn"].apply(lambda s: 1 if s == "Yes" else 0)

    MultipleLines = pd.get_dummies(df["MultipleLines"], prefix="MultipleLines")
    df.drop("MultipleLines", inplace=True, axis=1)
    df = df.join(MultipleLines)
    InternetService = pd.get_dummies(df["InternetService"], prefix="InternetService")
    df.drop("InternetService", inplace=True, axis=1)
    df = df.join(InternetService)
    OnlineSecurity = pd.get_dummies(df["OnlineSecurity"], prefix="OnlineSecurity")
    df.drop("OnlineSecurity", inplace=True, axis=1)
    df = df.join(OnlineSecurity)
    OnlineBackup = pd.get_dummies(df["OnlineBackup"], prefix="OnlineBackup")
    df.drop("OnlineBackup", inplace=True, axis=1)
    df = df.join(OnlineBackup)
    DeviceProtection = pd.get_dummies(df["DeviceProtection"], prefix="DeviceProtection")
    df.drop("DeviceProtection", inplace=True, axis=1)
    df = df.join(DeviceProtection)
    TechSupport = pd.get_dummies(df["TechSupport"], prefix="TechSupport")
    df.drop("TechSupport", inplace=True, axis=1)
    df = df.join(TechSupport)
    StreamingTV = pd.get_dummies(df["StreamingTV"], prefix="StreamingTV")
    df.drop("StreamingTV", inplace=True, axis=1)
    df = df.join(StreamingTV)
    StreamingMovies = pd.get_dummies(df["StreamingMovies"], prefix="StreamingMovies")
    df.drop("StreamingMovies", inplace=True, axis=1)
    df = df.join(StreamingMovies)
    Contract = pd.get_dummies(df["Contract"], prefix="Contract")
    df.drop("Contract", inplace=True, axis=1)
    df = df.join(Contract)
    PaymentMethod = pd.get_dummies(df["PaymentMethod"], prefix="PaymentMethod")
    df.drop("PaymentMethod", inplace=True, axis=1)
    df = df.join(PaymentMethod)
    y_df = df.pop("Churn")
    # x_df = df.drop("Churn", inplace=True, axis=1)

    return df, y_df

df= pd.read_csv("https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv")


# replace blanks with np.nan
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
# convert to float64
df["TotalCharges"] = df["TotalCharges"].astype("float64")

imputer = SimpleImputer(strategy='median')
df['TotalCharges'] = imputer.fit_transform(df['TotalCharges']).reshape(1, -1)

x, y = clean_data(df)

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--solver', type=str, default='lbfgs', help="Algorithm to use in the optimization problem.")
    parser.add_argument('--penalty', type=str, default='l2', help="Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.")
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))
    run.log("Solver:", str(args.solver))
    run.log("Penalty:", str(args.penalty))

    model = LogisticRegression(penalty=args.penalty, C=args.C, max_iter=args.max_iter, solver=args.solver)
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))

if __name__ == "__main__":
    main()
