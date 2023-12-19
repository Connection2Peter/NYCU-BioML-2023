import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class Roc_curve:
    def __init__(self, smooth=10):
        self.titles  = []
        self.aucs    = []
        self.folds   = 0
        self.tprs    = []
        self.fprs    = []
        self.y_probs = np.array([])
        self.y_tests = np.array([])
    
    def append(self, y_test, y_prob):
        self.folds += 1
        self.y_tests = np.append(self.y_tests, y_test)
        self.y_probs = np.append(self.y_probs, y_prob)


    def add(self, title, y_test=None, y_probs=None):
        if self.folds == 0:
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc     = auc(fpr, tpr)

            self.titles.append(title)
            self.aucs.append(roc_auc)
            self.fprs.append(fpr)
            self.tprs.append(tpr)

            self.y_probs = np.array([])
            self.y_tests = np.array([])

        else:
            fpr, tpr, _ = roc_curve(self.y_tests, self.y_probs)
            roc_auc     = auc(fpr, tpr)

            self.titles.append(title)
            self.aucs.append(roc_auc)
            self.fprs.append(fpr)
            self.tprs.append(tpr)

            self.folds   = 0
            self.y_probs = np.array([])
            self.y_tests = np.array([])


    def plot(self, output=None):
        for i in range(len(self.fprs)):
            plt.plot(self.fprs[i], self.tprs[i], lw=2, label=f'{self.titles[i]} (AUC = {self.aucs[i]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        if output == None:
            plt.show()
        else:
            plt.savefig(output, dpi=600)
