# Comparison of performance between AutoML and hyperdrive from Azure

Generally, telecom companies focus on attracting new clients while at the same time forgetting to retain them. According to this article, increasing customer retention by 5% can increase profits by 25% to 95%.

When customers stop doing business with a company or a particular service, it is called churn. Also, it is known as customer attrition. Hence, it is possible to increase profits by taking actions that are conducive to retaining customers.

Therefore, the goal of the present project is to develop the best model possible using two services from Microsoft Azure: AutoML and Hyperdrive. AutoML is the process of automating the time-consuming, iterative tasks of machine learning model development. Hyperdrive is an Azure Service that is responsible for executing the search for previously defined hyperparameters within a search space, in a way that minimizes the error between the prediction and the actual values.

https://towardsdatascience.com/customer-churn-in-telecom-segment-5e49356f39e5
https://towardsdatascience.com/machine-learning-case-study-telco-customer-churn-prediction-bc4be03c9e1d
https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml


## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
The data set for this classification problem is taken from Kaggle and stems from the IBM sample dataset collection.

https://www.kaggle.com/blastchar/telco-customer-churn

### Task

This is a classification problem with an added difficulty: this is a dataset for which the target variable is highly unbalanced.

The dataset contains all the data to predict the behavior to retain customers. Each row represents a customer. Each column contains the customer's attributes. The datasets include information about customers who left within the last month in a column called Churn; services that each customer has signed up for like phone, multiple lines, internet, online security, and others; information about the customer like how long they have been a customer, contract, payment method, and others; demographic information about customers like gender, age range, and if they have partners and dependents. This dataset contains about 7043 unique values and 21 columns.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

The dataset is accessible through the following link:

- https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
