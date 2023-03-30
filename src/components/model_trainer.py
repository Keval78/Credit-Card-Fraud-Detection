import os
import sys
import math
import time
from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    KFold, 
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict
)

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, roc_curve,
    accuracy_score, 
    classification_report, 
    make_scorer, 
    matthews_corrcoef, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

from imblearn.pipeline import Pipeline, make_pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    trained_models_folder_path = os.path.join("artifacts", "models")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def LogisiticRegression(self, X_train, y_train): # Logistic Regression
        grid_log_reg = GridSearchCV(LogisticRegression(), {'C': [1000], 'penalty': ['l2']})
        grid_log_reg.fit(X_train, y_train)
        return grid_log_reg.best_estimator_

    def KNearest(self, X_train, y_train): # KNN Classifier
        grid_knears = GridSearchCV(KNeighborsClassifier(), {'algorithm': ['auto'], 'n_neighbors': [2]})
        grid_knears.fit(X_train, y_train)
        return grid_knears.best_estimator_

    def SVC(self, X_train, y_train): # Support Vector Classifier
        grid_svc = GridSearchCV(SVC(), {'C': [0.9], 'kernel': ['rbf']})
        grid_svc.fit(X_train, y_train)
        return grid_svc.best_estimator_

    def DecisionTree(self, X_train, y_train): # DecisionTree Classifier
        grid_tree = GridSearchCV(DecisionTreeClassifier(), {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4)), "min_samples_leaf": list(range(5,7))})
        grid_tree.fit(X_train, y_train)
        return grid_tree.best_estimator_

    def RandomForest(self, X_train, y_train): # Random Forest Classifier
        pipeline_rf = Pipeline([('model', RandomForestClassifier(n_jobs=-1, random_state=1))])
        rf_params = {'model__n_estimators': [75]}
        MCC_scorer = make_scorer(matthews_corrcoef)
        sss = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        grid_rf = GridSearchCV(estimator=pipeline_rf, param_grid=rf_params, scoring=MCC_scorer, cv=sss, n_jobs=-1)
        grid_rf.fit(X_train, y_train)
        return grid_rf.best_estimator_
    
    def XGBoost(self, X_train, y_train): # XGBoost Classifier
        model = XGBClassifier(random_state=1)    
        xgb_param = {'learning_rate': [0.2], 'max_depth': [2], 'n_estimators': [100]}
        MCC_scorer = make_scorer(matthews_corrcoef)
        sss = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        grid_xgb = GridSearchCV(estimator=model, param_grid=xgb_param, scoring=MCC_scorer, cv=sss, n_jobs=-1)
        grid_xgb.fit(X_train, y_train)
        return grid_xgb.best_estimator_

    def model_evaluation(self, models, X_test, y_test):
        # Defining number of columns
        n_cols, n_models = 3, len(models)
        n_rows = math.ceil(n_models/n_cols)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(21, 7*n_rows))
        i, j = 0, 0
        for model in models:
            y_pred = models[model]['output'].predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp = disp.plot(include_values=True, cmap='Blues', ax=ax[i, j])
            disp.ax_.set_title('Confusion Matrix')

            # Generate report
            report = classification_report(y_test, y_pred)

            ax[i, j].axis('off')
            ax[i, j].set_title(f'{model} Classification Report')
            ax[i, j].annotate(report, xy=(0.1, 0), xytext=(0, -10), xycoords='axes fraction', textcoords='offset pixels', va='top')

            if j==n_cols-1: j, i = 0, i+1
            else: j+=1

        plt.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join('artifacts', "ConfusionMatrix.png"))
        # plt.show()
    
    def roc_curves(self, models, X_test, y_test):
        #ROC Curve
        plt.figure(figsize=(16,8))
        plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
        for model in models:
            if hasattr(models[model]['output'],'decision_function'):
                probs=models[model]['output'].decision_function(X_test) 
            elif hasattr(models[model]['output'],'predict_proba'):
                probs=models[model]['output'].predict_proba(X_test) [:,1]
            log_fpr, log_tpr, log_thresold = roc_curve(y_test, probs)
            roc_auc_scor = roc_auc_score(y_test, probs)
            plt.plot(log_fpr, log_tpr, label='{} Score: {:.4f}'.format(model, roc_auc_scor))
            logging.info('ROC-AUC value for {} {:.4f}'.format(model, roc_auc_scor))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.01, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3), arrowprops=dict(facecolor='#6E726D', shrink=0.05))
        plt.legend()
        plt.savefig(os.path.join('artifacts', "ROCAUC.png"))
        # plt.show()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "LogisiticRegression": {'function': self.LogisiticRegression},
                "KNearest": {'function': self.KNearest},
                "SupportVectorClassifier": {'function': self.SVC},
                "DecisionTreeClassifier": {'function': self.DecisionTree},
                "RandomForestClassifier": {'function': self.RandomForest},
                "XGBoostClassifier": {'function': self.XGBoost}
            }

            for model in models:
                logging.info("Executing... " + model)
                start_time=time.time()
                models[model]['output'] = models[model]['function'](X_train, y_train)
                training_execution_time=time.time()-start_time
                models[model]['training_execution_time'] = training_execution_time
                logging.info(model + " Execution Complete...")
                save_object (
                    file_path = os.path.join (
                        self.model_trainer_config.trained_models_folder_path,
                        '{}.pkl'.format(model)
                    ),
                    obj = models[model]['output']
                )
            
            logging.info("Evaluating Models...")
            self.model_evaluation(models, X_test, y_test)
            
            logging.info("Calculating ROC-AUC values for Models...")
            self.roc_curves(models, X_test, y_test)

            return

        except Exception as e:
            raise CustomException(e, sys)
