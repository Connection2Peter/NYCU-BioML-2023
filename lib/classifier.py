##### Import
#import os, joblib
#from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier



##### Functions
def RandomForest(nTree):
	return RandomForestClassifier(n_estimators=nTree, random_state=87, n_jobs=-1)
	