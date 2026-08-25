# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_predict, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as dc
from sklearn.svm import SVC
import utils as u
from sklearn.neighbors import KNeighborsClassifier

dataset = load_digits() # carico il dataset

X = dataset['data']     # le feature (immagini 8x8 appiattite a 64 elementi)
y = dataset['target']   # le classi (numeri da 0 a 9)

# Definisco le griglie  Nested
param_grid_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
param_grid_svc = {'C': [1, 10, 100], 'kernel': ['linear']}
param_grid_knn = {'n_neighbors': [3, 5, 7]}

Alb_dec = dc()
svc = SVC() 
knn = KNeighborsClassifier()

# GridSearchCV (Ottimizzazione parametri)
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

# Stampa curva grafico
u.pr_curve(y, proba_dt, score_svc, proba_knn)

def stampa_risultati_cv(nome_modello, risultati):
    print("\n" + "="*50)
    print(f"  REPORT METRICHE ({nome_modello}) [Nested 5-Fold CV]")
    print("="*50)
    # Calcolo media e deviazione standard per ogni metrica
    print(f"Precision: {np.mean(risultati['test_precision']):.3f} (± {np.std(risultati['test_precision']):.3f})")
    print(f"Recall:    {np.mean(risultati['test_recall']):.3f} (± {np.std(risultati['test_recall']):.3f})")
    print(f"F1-Score:  {np.mean(risultati['test_f1']):.3f} (± {np.std(risultati['test_f1']):.3f})")

stampa_risultati_cv("KNN", risultati_knn)
stampa_risultati_cv("SVC", risultati_svc)
stampa_risultati_cv("ALBERO DI DECISIONE", risultati_dt)

# Addestramento finale
grid_Alb_dec.fit(X, y)
grid_svc.fit(X, y)
grid_knn.fit(X, y)

print("\n" + "="*50)
print("  PARAMETRI CALCOLATI DALLA GRIDSEARCHCV")
print("="*50)
print("Albero di Decisione:", grid_Alb_dec.best_params_)
print("SVC:", grid_svc.best_params_)
print("KNN:", grid_knn.best_params_)

indice = random.randint(0, len(X) - 1)

print("\nIL VERO NUMERO DA PREDIRE E': ", y[indice])

# Test sull numero da predire
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

input("\n[Premi INVIO per chiudere il grafico e terminare...]")