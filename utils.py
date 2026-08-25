# -*- coding: utf-8 -*-
import pylab as pl
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.datasets import load_digits

# mappatura dei nomi dei target
_target_names = load_digits().target_names

def category_name(nuovo_test):  
       return _target_names[int(nuovo_test)]        

def pr_curve(y_test, proba_dt, score_svc, proba_knn):
   
    y_bin = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9])
    
    y_bin_flat = y_bin.ravel()
    
    # Calcolo PR per ogni predizione
    precision, recall, _ = precision_recall_curve(y_bin_flat, proba_dt.ravel())
    precision_SV, recall_SV, _ = precision_recall_curve(y_bin_flat, score_svc.ravel())
    precision_knn, recall_knn, _ = precision_recall_curve(y_bin_flat, proba_knn.ravel())
    
    area = auc(recall, precision)
    area_sv = auc(recall_SV, precision_SV)
    area_k = auc(recall_knn, precision_knn)
    
    print("\n" + "="*50)
    print("      AREA SOTTO LA CURVA PRECISION-RECALL (AUC)")
    print("="*50)
    print("Area Under Curve (Decision Tree): %0.3f" % area)
    print("Area Under Curve (SVM):           %0.3f" % area_sv)
    print("Area Under Curve (KNN):           %0.3f" % area_k)
    
    pl.ion()
    pl.clf()
    pl.plot(recall, precision, label=f"Decision Tree (AUC = {area:.2f})")
    pl.plot(recall_SV, precision_SV, label=f"Support Vector Machine (AUC = {area_sv:.2f})")
    pl.plot(recall_knn, precision_knn, label=f"KNN (AUC = {area_k:.2f})")
    pl.xlabel("Recall")
    pl.ylabel("Precision")
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title("Curva Precision-Recall (Micro-Averaged)")
    pl.legend(loc="lower left")
    pl.draw()
    pl.pause(0.1)