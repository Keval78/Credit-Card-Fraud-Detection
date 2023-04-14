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

from plotly import (
    graph_objs as go,
    express as px,
    figure_factory as ff
)
from plotly.subplots import make_subplots

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
        class_, empty_class_ = ["Non-Fraud", "Fraud"], ["         ", "     "]
        # Defining number of columns
        n_cols, n_models = 3, len(models)
        n_rows = math.ceil(n_models/n_cols)
        
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=tuple(map(lambda x: f"<b>{x}</b>", models.keys())), vertical_spacing=0.1, horizontal_spacing=0.01)
        rows, cols, k = 1, 1, 1
        for model in models:
            y_pred = models[model]['output'].predict(X_test) # models[model]['predicitions']
            cm = confusion_matrix(y_test, y_pred)

            report = classification_report(y_test, y_pred)
            
            fig.add_trace(go.Heatmap(
                x=class_, y=class_ if cols==1 else empty_class_, z=cm, 
                colorscale='Greys', text = [["Non-Fraud"]*2,["Fraud"]*2], 
                zmin=0, zmax=100, hovertemplate = "True Class: %{text}<br>Predicted Class: %{x}<br>Count: %{z}",
                showscale=False), row=rows, col=cols)

            # Add annotations to first heatmap
            for i in range(len(class_)):
                for j in range(len(class_)):
                    color = "Black" if cm[j, i]<50 else "White"
                    fig.add_annotation(x=class_[i], y=class_[j] if cols==1 else empty_class_[j], text=f"<b>{cm[j, i]}<b>", font=dict(color=color, size=14), showarrow=False, xref=f"x{k}", yref=f"y{k}")

            fig.add_annotation(
                x=.5, y=-1.1,
                text="<b>"+report.split("accuracy")[0].replace("\n","<br>")+"</b>",
                font=dict(color='black', size=14),
                showarrow=False,
                xref=f"x{k}", yref=f"y{k}"
            )

            if cols==n_cols: cols, rows = 1, rows+1
            else: cols+=1
            k+=1

        fig.update_layout(title_text='<b>Confusion Matrices for each Machine Learning Model</b>', font=dict(size=16), height=1000, width=1400, showlegend=False)
        fig['data'][0]['showscale'] = True
        
        # create_and_download_obj("ConfusionMatrix", fig)
        save_object(
            file_path = os.path.join ('artifacts', 'charts', 'ConfusionMatrix'),
            obj = fig
        )
        # fig.show()
        # plt.savefig(os.path.join('artifacts', 'models',  "ConfusionMatrix.png"))
        # plt.show()
    
    def roc_curves(self, models, X_test, y_test):
        #ROC Curve
        fig = go.Figure()
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

        for model in models:
            if hasattr(models[model]['output'],'decision_function'):
                probs=models[model]['output'].decision_function(X_test)
            elif hasattr(models[model]['output'],'predict_proba'):
                probs=models[model]['output'].predict_proba(X_test) [:,1]
            fpr, tpr, log_thresold = roc_curve(y_test, probs)
            auc_score = roc_auc_score(y_test, probs)

            name = f"<b>{model} (AUC={auc_score:.5f})</b>"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=1200, height=1000,
            legend=dict(
                x=0.53,
                y=0.05,
                traceorder='normal',
                font=dict(family='sans-serif', size=15),
                bgcolor='White',
                bordercolor='Black',
                borderwidth=1,
                orientation="v"
            )
        )
        
        save_object(
            file_path = os.path.join ('artifacts', 'charts', 'ROCcurve'),
            obj = fig
        )

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
                        '{}'.format(model)
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
