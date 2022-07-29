import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


class PredMetrics:
    """class to investigate a classification model fitness"""
    def __init__(self, pred_pd, actual):
        self.df_compare = pd.DataFrame({'default_prob': pred_pd,
                                        'actual':  actual})
    
    def confusion_mat(self, threshold):
        """confusion matrix for a given threshold level"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        return pd.crosstab(classified_df.default_prob, classified_df.actual)

    def accuracy(self, threshold):
        """accuracy metric for a given threshold level - correct / total"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        correct = np.array(classified_df.default_prob == classified_df.actual).sum()
        return correct / classified_df.shape[0]
    
    def tpr(self, threshold):
        """true positive rate (aka recall/sensitivity) for a given threshold level - true positives / positives"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        tp = classified_df.default_prob[classified_df.actual == 1].sum()
        p = np.array(classified_df.actual == 1).sum()
        return tp/p

    def tnr(self, threshold):
        """true negative rate for a given threshold level - true negatives / negatives"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        tn = np.array(classified_df.default_prob[classified_df.actual == 0] == 0).sum()
        n = np.array(self.df_compare.actual == 0).sum()
        return tn/n

    def fpr(self, threshold):
        """true negative rate for a given threshold level - true negatives / negatives"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        fp = np.array(classified_df.default_prob[classified_df.actual == 0] == 1).sum()
        n = np.array(self.df_compare.actual == 0).sum()
        return fp/n
    
    def fnr(self, threshold):
        """false negative rate for a given threshold level - false negatives / positives"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        fn = np.array(classified_df.default_prob[classified_df.actual == 1] == 0).sum()
        p = np.array(classified_df.actual == 1).sum()
        return fn/p
    
    def balanced_acc(self, threshold):
        """balanced accuracy - (tpr + tnr)/n"""
        return (self.tpr(threshold) + self.tnr(threshold)) / 2

    def ppv(self, threshold):
        """positive predictive value (aka precision) for a given threshold"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        tp = classified_df.default_prob[classified_df.actual == 1].sum()
        pp = classified_df.default_prob.sum()
        return tp/pp
    
    def f1_score2(self, threshold):
        return 2*((self.ppv(threshold) * self.tpr(threshold))/(self.ppv(threshold) + self.tpr(threshold)))

    def f1_score(self, threshold):
        """F1 score - tp/(tp + 1/2(fp + fn))"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        tp = classified_df.default_prob[classified_df.actual == 1].sum()
        fp = np.array(classified_df.default_prob[classified_df.actual == 0] == 1).sum()
        fn = np.array(classified_df.default_prob[classified_df.actual == 1] == 0).sum()
        return (2*tp)/(2*tp + (fp + fn))

    def f1_score_sk(self, threshold, average):
        """sklearn function for troubleshooting low f1_score fo the models"""
        classified_df = self.df_compare.copy()
        classified_df['default_prob'] = np.where(self.df_compare['default_prob'] > threshold, 1, 0)
        return f1_score(classified_df.actual, classified_df.default_prob, average = average)

    def max_balanced_acc(self, granularity):
        """balanced accuracy - (tpr + tnr)/n"""
        thresholds = np.linspace(start=0, stop=1, num=granularity)
        balanced_acc_df = pd.DataFrame({'thresholds': thresholds,
                                        'tpr': [self.tpr(i) for i in thresholds],
                                        'tnr': [self.tnr(i) for i in thresholds]})
        balanced_acc_df['balanced_acc'] = (balanced_acc_df.tpr + balanced_acc_df.tnr)/2
        max_bacc = balanced_acc_df.iloc[np.argmax(balanced_acc_df.balanced_acc),:]
        return max_bacc