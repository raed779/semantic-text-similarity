import warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import streamlit
from tabulate import tabulate

from pandas import read_csv
from numpy import argmax
from numpy import vstack
import seaborn as sns
from matplotlib import pyplot
from sklearn.metrics import classification_report, precision_recall_curve, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def warn(*args, **kwargs):
    pass


warnings.warn = warn
# Utils


def label_encode_row(x):
    """Label encodes values. 

    Args:
        x (pd.DataFrame): dataframe to label encode
    Returns:
        x (pd.DataFrame): label encoded dataframe                                         

    """
    x, _ = pd.factorize(x)
    return x


def transform_data(df, columns_to_Encod):
    """Function to transform data.
    Args:
        df (pd.DataFrame):  input to transform                            
        columns_to_Encod (list) : columns to encode                      
    Returns:
        df_all_Feature (pd.DataFrame): data  transformed
    """
    df_ = df.copy()
    Cols = ['label_pred_1', 'label_pred_2',
            'label_pred_3', 'label_pred_4', 'label_pred_5']
    df_['label_true_original'] = df_['label_true']

    df_[columns_to_Encod] = pd.DataFrame(df_[columns_to_Encod].apply(
        label_encode_row, axis=1).to_list(), columns=columns_to_Encod)
    df_['label_true'] = df_['label_true'].mask((df_[Cols].values != df_[
                                               ['label_true']].values).all(axis=1).astype(int) == 1, 5, axis=0)
    df_dummies = pd.get_dummies(
        df_[columns_to_Encod[:-1]], columns=columns_to_Encod[:-1])
    df_all_Feature = pd.concat([df_, df_dummies], axis=1)

    return df_all_Feature


def test_transform_data(columns_to_Encod):
    """This function return an example of the transformation.
    Args:
        columns_to_Encod (pd.DataFrame): input
    """
    test_frame = pd.DataFrame(
        data=[df_corpus_top5.iloc[0].values], columns=df_corpus_top5.columns)
    table_input = tabulate(
        test_frame, headers=test_frame.columns, tablefmt='github')
    print("\033[0;32;47m    Input  :  \n")
    print("\033[0;32;47m " + table_input)
    result = transform_data(test_frame, columns_to_Encod)
    print("\n")
    print("\033[0;32;47m  Outout  :  \n")
    table_output = tabulate(result, headers=result.columns, tablefmt='github')
    print("\033[0;32;47m "+table_output)


def model_selection(X, y, models):
    """This function give an idea about the general performance of each model. 
       using K-Folds cross-validator (Split dataset into k consecutive folds).
       Each fold is  used once as a validation while the k - 1 remaining folds form the training set.
    Args:
        X (pd.DataFrame): X_data
        y (pd.DataFrame): y_data
        models (dict): SKlearn models

    Returns:
        df_evaluation (pd.DataFrame): data frame contains all scores
    """
    iteration = 0
    k = 7
    models_score = {
        "K_nn": [],
        "Naive_Bayes": [],
        "Logistic_Regression": [],
        "SVM": [],
        "Random_Forest": [],
        "Ada_Boost": [],
        'Xg_Boost': []
    }

    kf = KFold(n_splits=k)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        models["K_nn"].fit(X_train, y_train)
        models_score['K_nn'].append(models["K_nn"].score(X_test, y_test))

        models["Naive_Bayes"].fit(X_train, y_train)
        models_score['Naive_Bayes'].append(
            models["Naive_Bayes"].score(X_test, y_test))

        models["Logistic_Regression"].fit(X_train, y_train)
        models_score['Logistic_Regression'].append(
            models["Logistic_Regression"].score(X_test, y_test))

        models["SVM"].fit(X_train, y_train)
        models_score['SVM'].append(models["SVM"].score(X_test, y_test))

        models["Random_Forest"].fit(X_train, y_train)
        models_score['Random_Forest'].append(
            models["Random_Forest"].score(X_test, y_test))

        models["Ada_Boost"].fit(X_train, y_train)
        models_score['Ada_Boost'].append(
            models["Ada_Boost"].score(X_test, y_test))

        models["Xg_Boost"].fit(X_train, y_train)
        models_score['Xg_Boost'].append(
            models["Xg_Boost"].score(X_test, y_test))

    model_evaluation_dict = {}
    for model in models_score.keys():
        model_evaluation_dict[model] = [np.mean(models_score[model])]
    df_evaluation = pd.DataFrame.from_dict(model_evaluation_dict)
    df_evaluation = df_evaluation.rename(index={0: 'Accuracy'})
    return df_evaluation


def benchmark_ML_models(X_train, X_test, y_train, y_test, models, data):
    """All models predictions without hyperparameters optimization.

    Args:
        X_train (pd.DataFrame): X_train
        X_test (pd.DataFrame): X_test
        y_train (pd.DataFrame): y_train
        y_test (pd.DataFrame): y_test
        models (dict): SKlearn models
        data (pd.DataFrame): the test set

    Returns:
        data: the test set with all models predictions and scores
    """

    models["K_nn"].fit(X_train, y_train)
    predKNeighbors = models["K_nn"].predict(X_test)
    probability_KNeighbors = models["K_nn"].predict_proba(X_test)
    probability_KNeighbors_max = np.max(probability_KNeighbors, axis=1)

    models["Naive_Bayes"].fit(X_train, y_train)
    prednaive_b = models["Naive_Bayes"].predict(X_test)
    probability_naive_b = models["Naive_Bayes"].predict_proba(X_test)
    probability_naive_b_max = np.max(probability_naive_b, axis=1)

    models["Logistic_Regression"].fit(X_train, y_train)
    predLogistic_Regression = models["Logistic_Regression"].predict(X_test)
    probability_Logistic_Regression = models["Logistic_Regression"].predict_proba(
        X_test)
    probability_Logistic_Regression_max = np.max(
        probability_Logistic_Regression, axis=1)

    models["SVM"].fit(X_train, y_train)
    predSVM = models["SVM"].predict(X_test)
    probability_SVM = models["SVM"].predict_proba(X_test)
    probability_SVM_max = np.max(probability_SVM, axis=1)

    models["Random_Forest"].fit(X_train, y_train)
    predRandom_Forest = models["Random_Forest"].predict(X_test)
    probability_Random_Forest = models["Random_Forest"].predict_proba(X_test)
    probability_Random_Forest_max = np.max(probability_Random_Forest, axis=1)

    models["Ada_Boost"].fit(X_train, y_train)
    predada_b = models["Ada_Boost"].predict(X_test)
    probability_ada_b = models["Ada_Boost"].predict_proba(X_test)
    probability_ada_b_max = np.max(probability_ada_b, axis=1)

    models["Xg_Boost"].fit(X_train, y_train)
    predXg_Boost = models["Xg_Boost"].predict(X_test)
    probability_Xg_Boost = models["Xg_Boost"].predict_proba(X_test)
    probability_Xg_Boost_max = np.max(probability_Xg_Boost, axis=1)

    prediction = {
        "K_nn": predKNeighbors,
        "Naive_Bayes": prednaive_b,
        "Logistic_Regression": predLogistic_Regression,
        "SVM": predSVM,
        "Random_Forest": predRandom_Forest,
        "Ada_Boost": predada_b,
        'Xg_Boost': predXg_Boost
    }

    probability = {
        "K_nn": probability_KNeighbors_max,
        "Naive_Bayes": probability_naive_b_max,
        "Logistic_Regression": probability_Logistic_Regression_max,
        "SVM": probability_SVM_max,
        "Random_Forest": probability_Random_Forest_max,
        "Ada_Boost": probability_ada_b_max,
        'Xg_Boost': probability_Xg_Boost_max
    }
    for i in models:
        data[i+"_prediction"] = prediction[i]
        data[i+"_score"] = probability[i]

    return data


KNeighbors = KNeighborsClassifier(n_neighbors=4)
Gaussian = GaussianNB()
Lr = LogisticRegression()
SVM = svm.SVC(probability=True)
RandomForest = RandomForestClassifier()
AdaBoost = AdaBoostClassifier(n_estimators=20)
XGB = xgb.XGBClassifier()

models = {
    "K_nn": KNeighbors,
    "Naive_Bayes": Gaussian,
    "Logistic_Regression": Lr,
    "SVM": SVM,
    "Random_Forest": RandomForest,
    "Ada_Boost": AdaBoost,
    "Xg_Boost": XGB
}
# Load data
# Feature engineering


"""
df_corpus_top5 = pd.read_csv('../data/df_corpus_top5.csv',sep=',')
df_corpus_top5 = df_corpus_top5.drop(columns=["Unnamed: 0"])
df_corpus_top5.head(3)
df_test_top5 = pd.read_csv('../data/df_test_top5.csv',sep=',')
df_test_top5 = df_test_top5.drop(columns=["Unnamed: 0"])
df_test_top5.head(3)
columns_to_encod = ['label_pred_1',
                   'label_pred_2',
                   'label_pred_3',
                   'label_pred_4',
                   'label_pred_5',
                   'label_true',]
test_transform_data(columns_to_encod)
df_train = transform_data(df_corpus_top5, columns_to_encod)
df_test = transform_data(df_test_top5, columns_to_encod)
score_features = [ 'score_1', 'score_2', 'score_3', 'score_4', 'score_5']
label_pred_features = ['label_pred_1','label_pred_2', 'label_pred_3', 'label_pred_4', 'label_pred_5']
is_pred_correct_features = ['is_pred_1_correct', 'is_pred_2_correct','is_pred_3_correct','is_pred_4_correct', 'is_pred_5_correct','is_top3_correct','is_top5_correct',]
label_pred_dummies_features = ['label_pred_1_0','label_pred_2_0', 'label_pred_2_1', 'label_pred_3_0', 'label_pred_3_1','label_pred_3_2','label_pred_4_0', 'label_pred_4_1','label_pred_4_2','label_pred_4_3', 'label_pred_5_0', 'label_pred_5_1', 'label_pred_5_2','label_pred_5_3', 'label_pred_5_4']
X_train = df_train[ score_features + label_pred_features + label_pred_dummies_features  ]
y_train = df_train[['label_true']]   

X_test = df_test[ score_features + label_pred_features + label_pred_dummies_features ]
y_test = df_test[['label_true']]     
# Modeling
### Opted for the following classification models:
- Naive Bayes
- KNN
- Logistic Regression
- Random Forest
- SVM
- Ada-Boost
- XG-Boost
## As a method for estimating the reliability of machine learning models, we used K-Fold Cross Validation
# features used : All of them except  is_pred_K_correct
model_selection(X_train, y_train, models)
# features used : only score
model_selection(X_train, y_train, models)
#  features used : all 
"""
