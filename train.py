import platform

print(platform.platform())
import sys

print("Python", sys.version)
import numpy

print("NumPy", numpy.__version__)
import scipy

print("SciPy", scipy.__version__)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump
from sklearn import preprocessing
import dvc.api
import mlflow
import mlflow.sklearn


mlflow.set_experiment("SmartAD")

def train():

    # Load directory paths for persisting model

    MODEL_DIR = os.path.join(os.getcwd(), "models")
    MODEL_PATH_LRM = os.path.join(MODEL_DIR, "clf_lrm.joblib")
    MODEL_PATH_NN = os.path.join(MODEL_DIR, "clf_nn.joblib")
    
    #Load DataSets
    data_url= dvc.api.get_url(
    "data/browser_clean_data.csv",
    repo="https://github.com/abu-bakarr/10_Academy_Week2_Group",)
    

    
    df = pd.read_csv(data_url)
    mlflow.log_param("data_url", data_url)
    mlflow.log_param("input_rows", df.shape[0])
    mlflow.log_param("input_cols", df.shape[1])
    df.drop(["auction_id", "Unnamed: 0"], axis=1, inplace=True)
    print("Successfully loaded data")
    
    print("Label Encoding and Splitting data into train and test")
    cols = df.columns.to_list()
    for e in cols:
        df[e] = preprocessing.LabelEncoder().fit_transform(df[e])

    X1 = df.drop("user_response", axis=1) #prediction features
    y1 = df["user_response"]
        
    X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
    print("Shape of the testing data")
    print(X_test.shape)
    print(y_test.shape)
    

    # Models training



    # Logistic Regression Model Training
    logreg=LogisticRegression(random_state=None)
    # logreg.fit(trainX,y_train) 
    logreg.fit(X_train,y_train) 
    
    print("Logistic Regression Model trained")
    
    print("Accuracy Score",logreg.score(X_test, y_test))
    
    
    
    dump(logreg, MODEL_PATH_LRM)
    
    print("Logistic Regression Model saved to", MODEL_PATH_LRM)
    
    # model = LogisticRegression(random_state=None)

    # solvers = ["newton-cg", "lbfgs", "liblinear"]
    # penalty = ["l2"]
    # c_values = [100, 10, 1.0, 0.1, 0.01]
    # # define grid search
    # grid = dict(solver=solvers, penalty=penalty, C=c_values)
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # grid_search = GridSearchCV(
    #     estimator=model,
    #     param_grid=grid,
    #     n_jobs=-1,
    #     cv=cv,
    #     scoring="accuracy",
    #     error_score=0,
    # )
    # grid_result = grid_search.fit(X_train, y_train)
    
    # print("[INFO] evaluating...")
    # bestModel = grid_result.best_estimator_
    # print("R2: {:.5f}".format(bestModel.score(X_test, y_test)))


if __name__ == "__main__":
    train()
