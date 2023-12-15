import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class Roc_curve:
    def __init__(self, smooth=100):
        self.titles    = []
        self.aucs      = []
        self.folds     = 0
        self.mean_tpr  = 0.0
        self.mean_fpr  = np.linspace(0, 1, smooth)
        self.mean_tprs = []
        self.mean_fprs = []
    
    def append(self, y_test, y_probs):
        self.folds += 1
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        self.mean_tpr += np.interp(self.mean_fpr, fpr, tpr)
        self.mean_tpr[0] = 0.0


    def add(self, title, y_test=None, y_probs=None):
        if self.folds == 0:
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            mean_tpr   += np.interp(self.mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

            self.titles.append(title)
            self.aucs.append(auc(self.mean_fpr, mean_tpr))
            self.mean_fprs.append(self.mean_fpr)
            self.mean_tprs.append(self.mean_tpr)

        else:
            self.mean_tpr /= self.folds

            self.folds = 0

            self.titles.append(title)
            self.aucs.append(auc(self.mean_fpr, self.mean_tpr))

            self.mean_fprs.append(self.mean_fpr)
            self.mean_tprs.append(self.mean_tpr)

            self.mean_tpr = 0.0


    def plot(self):
        for i in range(len(self.mean_fprs)):
            plt.plot(self.mean_fprs[i], self.mean_tprs[i], lw=2, label=f'{self.titles[i]} (AUC = {self.aucs[i]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Versicolor')
        plt.legend(loc='lower right')
        plt.show()