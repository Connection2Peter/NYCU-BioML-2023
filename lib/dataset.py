##### Import
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold



##### Functions
def Balance(db1, db2):
	num1, num2 = len(db1), len(db2)

	if num1 > num2:
		return [random.sample(db1, num2), db2]

	return [db1, random.sample(db2, num1)]

def SplitNfold(numSplit):
	return StratifiedKFold(n_splits=numSplit, shuffle=True, random_state=87)


def Evaluation(y_test, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

	Metrics = [
		tp / (tp + fn),
		tn / (tn + fp),
		accuracy_score(y_test, y_pred),
		matthews_corrcoef(y_test, y_pred),
		roc_auc_score(y_test, y_pred),
	]

	return Metrics
