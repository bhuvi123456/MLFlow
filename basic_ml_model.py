import mlflow         #To track experiments we use mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import os
#I can solve this problem with both regression as well as classification 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import argparse

def get_data():
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        return pd.read_csv(url,sep=';')
    except Exception as e:
        raise e
    
def evaluate_model(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    confusion = confusion_matrix(y_test,y_pred)
    classification = classification_report(y_test,y_pred)
    return accuracy,confusion,classification



def main(model_type='randomforest'):
    df = get_data()
    train,test = train_test_split(df)
    x_train = train.drop(columns = 'quality')
    x_test = test.drop(columns = 'quality')
    y_train = train['quality']
    y_test = test['quality']
    with mlflow.start_run:
        if model_type ==  'randomforest':
            model = RandomForestClassifier()
            params = {
                'n_estimators' : [100,150,250,300],
                'max_depth' : [5,10,15,20],
            }
        elif model_type == 'adaboost':
            model = AdaBoostClassifier()
            params = {
                'n_estimators': [100, 150, 250, 300],
                'learning_rate' : [0.01,0.1,0.5,1]
            }
        grid = GridSearchCV(estimator = model,param_grid = params)
        grid.fit(x_train,y_train)
        y_pred = grid.predict(x_test)
        best_params = grid.best_params_
        accuracy,confusion,classification = evaluate_model(y_test,y_pred)
        print(f"Best parameters: {grid.best_params_}")
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{confusion}")
        print(f"Classification Report:\n{classification}")
        mlflow.log_param("n_estimators",best_params['n_estimators'])
        mlflow.log_param("max_depth",best_params['max_depth'])
        mlflow.metrics("accuracy", accuracy)
        mlflow.metrics("confusion_matrix",confusion)


if __name__ == '__main__':
    '''args = argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n",default=50,type = int)
    args.add_argument("--max_depth","-m",default=5,type = int)
    parse_args = args.parse_args()'''
    try:
        main()
    except Exception as e:
        raise e
#Using this argsparse instead of using gridsearch cv we can do hyperparameter in cmd itself but using gridsearchcv
#  is a better option