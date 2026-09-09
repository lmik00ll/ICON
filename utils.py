# -*- coding: utf-8 -*-
import pylab as pl
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.datasets import load_digits
import heapq
import numpy as np

_target_names = load_digits().target_names

def category_name(nuovo_test):  
       return _target_names[int(nuovo_test)]        

def pr_curve(y_test, proba_dt, score_svc, proba_knn):
   
    y_bin = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9])
    
    y_bin_flat = y_bin.ravel()
    
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

def astar_decision_tree(tree_model, target_class):
    tree_ = tree_model.tree_
    open_set = []
    heapq.heappush(open_set, (tree_.impurity[0], 0, 0, [0]))
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        
        is_leaf = tree_.children_left[current] < 0 and tree_.children_right[current] < 0
        
        if is_leaf:
            pred = np.argmax(tree_.value[current])
            if pred == target_class:
                return path, g
        else:
            left = tree_.children_left[current]
            right = tree_.children_right[current]
            
            for child in [left, right]:
                if child >= 0:
                    g_new = g + 1
                    h_new = tree_.impurity[child]
                    f_new = g_new + h_new
                    heapq.heappush(open_set, (f_new, g_new, child, path + [child]))
                    
    return None, -1