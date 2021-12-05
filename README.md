# Comparison of performance between AutoML and hyperdrive from Azure

Generally, telecom companies focus on attracting new clients while at the same time forgetting to retain them. According to this article, increasing customer retention by 5% can increase profits by 25% to 95%.

When customers stop doing business with a company or a particular service, it is called churn. Also, it is known as customer attrition. Hence, it is possible to increase profits by taking actions that are conducive to retaining customers.

Therefore, the goal of the present project is to develop the best model possible using two services from Microsoft Azure: AutoML and Hyperdrive. AutoML is the process of automating the time-consuming, iterative tasks of machine learning model development. Hyperdrive is an Azure Service that is responsible for executing the search for previously defined hyperparameters within a search space, in a way that minimizes the error between the prediction and the actual values.

https://towardsdatascience.com/customer-churn-in-telecom-segment-5e49356f39e5
https://towardsdatascience.com/machine-learning-case-study-telco-customer-churn-prediction-bc4be03c9e1d
https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml


## Project Set Up and Installation

In order to run this project, this project should be cloned using git repository in a Notebook Terminal. There are two jupyter notebooks: automl.ipynb and hyperparameter_tuning_forest.ipynb.

## Dataset

### Overview
The data set for this classification problem is taken from Kaggle and stems from the IBM sample dataset collection.

https://www.kaggle.com/blastchar/telco-customer-churn

This dataset contains a total of 7043 customers and 21 attributes, coming from personal characteristics, service signatures, and contract details. About 5174 customers are active, and 1869 are churned. The target variable for this dataset is the Churn feature.


### Task

This is a classification problem with an added difficulty: this is a dataset for which the target variable is highly unbalanced.

The dataset contains all the data to predict the behavior to retain customers. Each row represents a customer. Each column contains the customer's attributes. The datasets include information about customers who left within the last month in a column called Churn; services that each customer has signed up for like phone, multiple lines, internet, online security, and others; information about the customer like how long they have been a customer, contract, payment method, and others; demographic information about customers like gender, age range, and if they have partners and dependents. This dataset contains about 7043 unique values and 21 columns.

The dataset was prepared in two ways. The first way was left as it came to feed the AutoML algorithm. This is, registering the dataset in the ML Workspace. The second way was to preprocess the dataset to feed the hyperdrive algorithm. This is done because there are multiple categorical variables, and they need to be encoded adequately to feed the scikit-learn algorithms.

The dataset was also split into train and test with 20% of its data as test data, with the option stratify as true because of the unbalanced proportion between churned and not churned clients.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

The dataset is accessible through the following link:

- https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

As stated earlier, this is a classification task with its output variable called Churn. 

The following lists was set to perform Automated ML:

- compute_target=cpu_cluster,
- task='classification',
- training_data=train_data,
- test_data = test_data,
- label_column_name='Churn',
- primary_metric='AUC_weighted',
- experiment_timeout_minutes=60,
- max_concurrent_iterations=5,
- max_cores_per_iteration=-1, 
- featurization= 'auto',

The main metric used for this experiment was AUC_weighted because accuracy is not the best metric for this task due to the unbalanced nature of the target variable.

The automl algorithm evaluate the model using the test data, and according to the higher AUC_weighted metric, the best is the model.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.


<img src="./images/architectural-diagram.png" width="400" height="600" 
     alt="Markdown" />
     
**Figure 1:** Architectured of the Solution.

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

