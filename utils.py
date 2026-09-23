# -*- coding: utf-8 -*-
import os
import pylab as pl
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
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

def stampa_matrice_confusione(y_true, y_pred, nome_modello):
    """
    Calcola e stampa a console la matrice di confusione multiclasse 10x10.
    """
    cm = confusion_matrix(y_true, y_pred)
    print("\n" + "="*50)
    print(f"  MATRICE DI CONFUSIONE - {nome_modello}")
    print("="*50)
    print("Righe: Cifra Reale (0-9) | Colonne: Cifra Predetta (0-9)\n")
    print(cm)

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

def estrai_kb_regole(tree_model, feature_names=None):
    """
    Attraversa l'albero con visita in profondità (DFS) ed estrae tutte
    le regole di produzione logiche (IF-THEN) della Knowledge Base.
    """
    tree_ = tree_model.tree_
    if feature_names is None:
        feature_names = [f"pixel_{i}" for i in range(tree_.n_features)]
    
    knowledge_base = []

    def dfs(node_id, percorso_corrente):
        is_leaf = tree_.children_left[node_id] < 0 and tree_.children_right[node_id] < 0
        
        if is_leaf:
            classe_predetta = int(np.argmax(tree_.value[node_id]))
            knowledge_base.append({
                "premesse": list(percorso_corrente),
                "conseguenza": classe_predetta
            })
            return

        feat = feature_names[tree_.feature[node_id]]
        thresh = tree_.threshold[node_id]

        dfs(tree_.children_left[node_id], percorso_corrente + [f"({feat} <= {thresh:.2f})"])
        dfs(tree_.children_right[node_id], percorso_corrente + [f"({feat} > {thresh:.2f})"])

    dfs(0, [])
    return knowledge_base

def salva_regole_su_file(kb, numero_predetto, cartella="kb"):
    os.makedirs(cartella, exist_ok=True)
    nome_file = os.path.join(cartella, f"regole_n_{numero_predetto}.txt")
    
    with open(nome_file, "w", encoding="utf-8") as f:
        for i, reg in enumerate(kb, start=1):
            premesse = " AND ".join(reg["premesse"])
            f.write(f"Regola {i:03d}: IF {premesse} THEN Cifra = {reg['conseguenza']}\n")
            
    return nome_file

def salva_fatti_su_file(istanza_fatti, target_reale, categoria_predetta, cartella="kb"):
    """
    Salva la descrizione assiomatica (fatti) del campione osservato
    in un file dedicato alla cifra predetta: fatti_n_{categoria_predetta}.txt
    """
    os.makedirs(cartella, exist_ok=True)
    nome_file = os.path.join(cartella, f"fatti_n_{categoria_predetta}.txt")
    
    with open(nome_file, "w", encoding="utf-8") as f:
        f.write(f"% FATTI OSSERVATI (ABox) - CAMPIONE CIFRA {categoria_predetta}\n")
        f.write(f"% Ground truth reale: {target_reale}\n\n")
        for i, val in enumerate(istanza_fatti):
            f.write(f"fatto(pixel_{i}, {val}).\n")
        f.write(f"\nground_truth_target({target_reale}).\n")
        
    return nome_file

def inferenza_kb(kb_regole, istanza_fatti):
    for idx, regola in enumerate(kb_regole, start=1):
        soddisfatta = True
        for premessa in regola["premesse"]:
            cond = premessa.strip("()")
            feat_name, op, val = cond.split()
            pixel_idx = int(feat_name.split("_")[1])
            soglia = float(val)
            valore_osservato = istanza_fatti[pixel_idx]

            if op == "<=" and not (valore_osservato <= soglia):
                soddisfatta = False
                break
            elif op == ">" and not (valore_osservato > soglia):
                soddisfatta = False
                break

        if soddisfatta:
            return regola["conseguenza"], idx, regola

    return None, -1, None