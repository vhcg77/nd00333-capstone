# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run


def clean_data(df):
    df.drop("customerID", axis=1, inplace=True)

    # replace blanks with np.nan
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    # convert to float64
    df["TotalCharges"] = df["TotalCharges"].astype("float64")

    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

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


df = pd.read_csv(
    "https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv"
)


x, y = clean_data(df)

# TODO: Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

run = Run.get_context()


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="The number of trees in the forest.",
    )
    parser.add_argument(
        "--max_features",
        type=float,
        default=1.0,
        help="The number of features to consider when looking for the best split.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
    )
    parser.add_argument(
        "--min_samples_split",
        type=float,
        default=1.0,
        help="The minimum number of samples required to split an internal node.",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=float,
        default=1.0,
        help="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
    )
    parser.add_argument(
        "--bootstrap",
        type=bool,
        default=True,
        help="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.",
    )

    args = parser.parse_args()

    run.log("Number of trees:", float(args.n_estimators))
    run.log("Max features:", int(args.max_features))
    run.log("Max depth:", str(args.max_depth))
    run.log("Min number of samples:", str(args.min_samples_split))
    run.log("Min number of samples at leaf node:", str(args.min_samples_leaf))
    run.log("Bootstrap:", str(args.bootstrap))

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        bootstrap=args.bootstrap,
    )
    model.fit(x_train, y_train)

    # accuracy = model.score(x_test, y_test)
    # run.log("Accuracy", float(accuracy))
    AUC_weighted = roc_auc_score(
        y_test, model.predict_proba(x_test)[:, 1], average="weighted"
    )
    run.log("AUC_weighted", np.float(AUC_weighted))

    os.makedirs("./outputs", exist_ok=True)
    joblib.dump(value=model, filename="./outputs/model.joblib")


if __name__ == "__main__":
    main()
