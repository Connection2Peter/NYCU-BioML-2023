##### Import
#import os, joblib
#from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



##### Functions
def DecisionTree():
    return DecisionTreeClassifier(random_state=87)

def RandomForest(nTree):
	return RandomForestClassifier(n_estimators=nTree, random_state=87, n_jobs=-1)
	