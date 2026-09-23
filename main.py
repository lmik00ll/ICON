# -*- coding: utf-8 -*-
import subprocess
import sys
import random
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_predict, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as dc
from sklearn.svm import SVC
import utils as u
from sklearn.neighbors import KNeighborsClassifier

dataset = load_digits() 

X = dataset['data']     
y = dataset['target']   

param_grid_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
param_grid_svc = {'C': [1, 10, 100], 'kernel': ['linear']}
param_grid_knn = {'n_neighbors': [3, 5, 7]}

Alb_dec = dc()
svc = SVC() 
knn = KNeighborsClassifier()

grid_Alb_dec = GridSearchCV(Alb_dec, param_grid_dt, cv=3)
grid_svc = GridSearchCV(svc, param_grid_svc, cv=3)
grid_knn = GridSearchCV(knn, param_grid_knn, cv=3)

print("Esecuzione della Nested Cross Validation in corso. Attendere...")

scoring = {'precision': 'precision_macro', 'recall': 'recall_macro', 'f1': 'f1_macro'}

risultati_dt = cross_validate(grid_Alb_dec, X, y, cv=5, scoring=scoring)
risultati_svc = cross_validate(grid_svc, X, y, cv=5, scoring=scoring)
risultati_knn = cross_validate(grid_knn, X, y, cv=5, scoring=scoring)

svc_curve = SVC(kernel='linear', C=10)
knn_curve = KNeighborsClassifier(3)
Alb_dec_curve = dc()

proba_dt = cross_val_predict(Alb_dec_curve, X, y, cv=5, method='predict_proba')
score_svc = cross_val_predict(svc_curve, X, y, cv=5, method='decision_function')
proba_knn = cross_val_predict(knn_curve, X, y, cv=5, method='predict_proba')

pred_dt_cm = cross_val_predict(Alb_dec_curve, X, y, cv=5)
u.stampa_matrice_confusione(y, pred_dt_cm, "Albero di Decisione")

u.pr_curve(y, proba_dt, score_svc, proba_knn)

def stampa_risultati_cv(nome_modello, risultati):
    print("\n" + "="*50)
    print(f"  REPORT METRICHE ({nome_modello}) [Nested 5-Fold CV]")
    print("="*50)
    print(f"Precision: {np.mean(risultati['test_precision']):.3f} (± {np.std(risultati['test_precision']):.3f})")
    print(f"Recall:    {np.mean(risultati['test_recall']):.3f} (± {np.std(risultati['test_recall']):.3f})")
    print(f"F1-Score:  {np.mean(risultati['test_f1']):.3f} (± {np.std(risultati['test_f1']):.3f})")

stampa_risultati_cv("KNN", risultati_knn)
stampa_risultati_cv("SVC", risultati_svc)
stampa_risultati_cv("ALBERO DI DECISIONE", risultati_dt)

grid_Alb_dec.fit(X, y)
grid_svc.fit(X, y)
grid_knn.fit(X, y)

print("\n" + "="*50)
print("  PARAMETRI CALCOLATI DALLA GRIDSEARCHCV")
print("="*50)
print("Albero di Decisione:", grid_Alb_dec.best_params_)
print("SVC:", grid_svc.best_params_)
print("KNN:", grid_knn.best_params_)

kb = u.estrai_kb_regole(grid_Alb_dec.best_estimator_)
print("\n" + "="*50)
print("  SINTESI KNOWLEDGE BASE (REGOLE DELL'ALBERO)")
print("="*50)
print(f"Numero totale di regole sintetizzate nella KB: {len(kb)}")

indice = random.randint(0, len(X) - 1)
target_reale = int(y[indice])

print("\nIL VERO NUMERO DA PREDIRE E': ", target_reale)

nuovo_numero = [X[indice]]

nuovo_test_a = grid_Alb_dec.predict(nuovo_numero)
nuovo_test_b = grid_svc.predict(nuovo_numero)
nuovo_test_c = grid_knn.predict(nuovo_numero)

categoria = u.category_name(nuovo_test_a[0])
categoria2 = u.category_name(nuovo_test_b[0])
categoria3 = u.category_name(nuovo_test_c[0])
 
print("\n" + "="*50)
print("            NUMERO PREDETTO ")
print("="*50)
print("Albero decisione: ", categoria, "\n\t\t   SVC: ", categoria2, "\n\t\t   Knn: ", categoria3)

file_regole = u.salva_regole_su_file(kb, categoria, cartella="kb")
file_fatti = u.salva_fatti_su_file(X[indice], target_reale, categoria, cartella="kb")
print(f"\n[OK] Tutte le regole salvate in: '{file_regole}'")
print(f"[OK] Fatti osservati del campione salvati in: '{file_fatti}'")

cifra_dedotta, id_regola, regola_attiva = u.inferenza_kb(kb, X[indice])
print("\n" + "="*50)
print("       INFERENZA DEDUTTIVA DALLA KNOWLEDGE BASE")
print("="*50)
print(f"Cifra dedotta formalmente: {cifra_dedotta}")
print(f"Regola attivata: #{id_regola}")
premesse_formali = " AND ".join(regola_attiva["premesse"])
print(f"Catena logica: IF {premesse_formali} THEN Cifra = {regola_attiva['conseguenza']}")

path, cost = u.astar_decision_tree(grid_Alb_dec.best_estimator_, target_reale)
print("\n" + "="*50)
print("      RICERCA A* NELL'ALBERO DI DECISIONE")
print("="*50)
print("Classe target cercata: ", target_reale)
print("Percorso nodi (radice -> foglia): ", path)
print("Costo del percorso (profondità): ", cost)

scelta = input("\nVuoi avviare l'estensione ontologica? (Y/N): ").strip().upper()
if scelta == 'Y':
    subprocess.run([sys.executable, "estensione_OWL.py"])