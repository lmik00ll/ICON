# -*- coding: utf-8 -*-
import os
import glob
import re
import numpy as np
from owlready2 import get_ontology, Thing, DataProperty, FunctionalProperty

def trova_ultimo_file_fatti(cartella="kb"):
    file_trovati = glob.glob(os.path.join(cartella, "fatti_n_*.txt"))
    if not file_trovati:
        return None
    file_trovati.sort(key=os.path.getmtime, reverse=True)
    return file_trovati[0]

def carica_fatti_da_file(percorso_file):
    pixel_valori = [0.0] * 64
    target_reale = None
    predizione_albero = None

    match_nome = re.search(r"fatti_n_(\d+)\.txt", os.path.basename(percorso_file))
    if match_nome:
        predizione_albero = int(match_nome.group(1))

    with open(percorso_file, "r", encoding="utf-8") as f:
        for riga in f:
            riga = riga.strip()
            match_pixel = re.match(r"fatto\(pixel_(\d+),\s*([0-9.]+)\)\.", riga)
            if match_pixel:
                idx = int(match_pixel.group(1))
                val = float(match_pixel.group(2))
                if 0 <= idx < 64:
                    pixel_valori[idx] = val
                continue
            
            match_target = re.match(r"ground_truth_target\((\d+)\)\.", riga)
            if match_target:
                target_reale = int(match_target.group(1))

    matrice_8x8 = np.array(pixel_valori).reshape((8, 8))
    return matrice_8x8, target_reale, predizione_albero

def estrai_caratteristiche_topologiche(immagine_8x8):
    soglia = np.mean(immagine_8x8)
    binaria = (immagine_8x8 > soglia).astype(int)
    
    cavita = 0
    for r in range(1, 7):
        for c in range(1, 7):
            if binaria[r, c] == 0:
                sopra = np.any(binaria[:r, c] == 1)
                sotto = np.any(binaria[r+1:, c] == 1)
                sinistra = np.any(binaria[r, :c] == 1)
                destra = np.any(binaria[r, c+1:] == 1)
                if sopra and sotto and sinistra and destra:
                    cavita += 1
    
    if cavita >= 4:
        num_anelli = 2
    elif cavita >= 1:
        num_anelli = 1
    else:
        num_anelli = 0

    colonne_attive = np.sum(binaria, axis=0)
    ha_tratto_verticale = bool(np.max(colonne_attive) >= 5)

    meta_sup = np.sum(binaria[:4, :])
    meta_inf = np.sum(binaria[4:, :])
    densita_inferiore = bool(meta_inf > (meta_sup * 1.2))

    return {
        "anelli_chiusi": num_anelli,
        "tratto_verticale": ha_tratto_verticale,
        "ha_base_pesante": densita_inferiore
    }

def valida_con_ontologia(proprieta_istanza, predizione_albero, cartella="kb"):
    os.makedirs(cartella, exist_ok=True)
    
    nome_file_specifico = f"ontologia_n_{predizione_albero}.owl"
    percorso_owl_specifico = os.path.join(cartella, nome_file_specifico)
    percorso_owl_base = os.path.join(cartella, "digits_bk.owl")
    
    percorso_assoluto_uri = "file://" + os.path.abspath(percorso_owl_specifico).replace("\\", "/")
    onto = get_ontology(percorso_assoluto_uri)

    with onto:
        class Cifra(Thing): pass
        class CifraConAnelli(Cifra): pass
        class CifraTrattoSemplice(Cifra): pass
        
        class numeroAnelli(DataProperty, FunctionalProperty):
            domain = [Cifra]
            range = [int]

        class haTrattoVerticale(DataProperty, FunctionalProperty):
            domain = [Cifra]
            range = [bool]

        class haBasePesante(DataProperty, FunctionalProperty):
            domain = [Cifra]
            range = [bool]

        class CifraOtto(CifraConAnelli): pass
        class CifraZero(CifraConAnelli): pass
        class CifraQuattro(CifraConAnelli): pass
        class CifraSei(CifraConAnelli): pass
        class CifraUno(CifraTrattoSemplice): pass

        nome_individuo = f"campione_cifra_{predizione_albero}"
        istanza_osservata = Cifra(nome_individuo)
        istanza_osservata.numeroAnelli = proprieta_istanza["anelli_chiusi"]
        istanza_osservata.haTrattoVerticale = proprieta_istanza["tratto_verticale"]
        istanza_osservata.haBasePesante = proprieta_istanza["ha_base_pesante"]

        classi_inferite = ["Cifra"]
        cifra_inferita_ontologia = None

        if istanza_osservata.numeroAnelli >= 2:
            istanza_osservata.is_a.append(CifraOtto)
            classi_inferite.append("CifraOtto")
            cifra_inferita_ontologia = 8
        elif istanza_osservata.haBasePesante and (istanza_osservata.haTrattoVerticale or istanza_osservata.numeroAnelli >= 1):
            istanza_osservata.is_a.append(CifraSei)
            classi_inferite.append("CifraSei")
            cifra_inferita_ontologia = 6
        elif istanza_osservata.numeroAnelli == 1 and istanza_osservata.haTrattoVerticale:
            istanza_osservata.is_a.append(CifraQuattro)
            classi_inferite.append("CifraQuattro")
            cifra_inferita_ontologia = 4
        elif istanza_osservata.numeroAnelli == 1 and not istanza_osservata.haTrattoVerticale:
            istanza_osservata.is_a.append(CifraZero)
            classi_inferite.append("CifraZero")
            cifra_inferita_ontologia = 0
        elif istanza_osservata.numeroAnelli == 0 and istanza_osservata.haTrattoVerticale and not istanza_osservata.haBasePesante:
            istanza_osservata.is_a.append(CifraUno)
            classi_inferite.append("CifraUno")
            cifra_inferita_ontologia = 1

    print("\n" + "="*55)
    print("      RAGIONAMENTO ONTOLOGICO FORMALE (OWL 2)")
    print("="*55)
    print("Fatti estratti dall'ABox:")
    print(f" - Individuo asserito:          {istanza_osservata.name}")
    print(f" - Numero anelli chiusi rilevati: {istanza_osservata.numeroAnelli}")
    print(f" - Presenza tratto verticale:    {istanza_osservata.haTrattoVerticale}")
    print(f" - Base con densita' maggiore:    {istanza_osservata.haBasePesante}")
    print(f"\nConcetti inferiti per il campione: {classi_inferite}")

    print("\n" + "-"*55)
    print("      DISAMBIGUAZIONE CONGIUNTA (DECISION TREE + ONTO-BK)")
    print("-"*55)
    print(f"Predizione candidata derivata da main.py: Cifra {predizione_albero}")
    
    if cifra_inferita_ontologia is not None:
        if cifra_inferita_ontologia == predizione_albero:
            print(f"Esito: L'ontologia CONFERMA formalmente la classe {predizione_albero} per coerenza assiomatica.")
        else:
            print(f"Esito: L'ontologia SEGNALA CONTRADDIZIONE: l'albero suggerisce Cifra {predizione_albero}, "
                  f"ma gli assiomi topologici dimostrano l'appartenenza a Cifra {cifra_inferita_ontologia}.")
    else:
        print(f"Esito: NEUTRO (Open World Assumption): la TBox non contiene vincoli sufficienti per validare o invalidare la Cifra {predizione_albero}.")

    onto.save(file=percorso_owl_specifico, format="rdfxml")
    onto.save(file=percorso_owl_base, format="rdfxml")
    print(f"\n[OK] Ontologia del campione salvata in: '{percorso_owl_specifico}'")
    print(f"[OK] Ontologia di riferimento aggiornata in: '{percorso_owl_base}'")

if __name__ == "__main__":
    ultimo_file = trova_ultimo_file_fatti("kb")
    
    if not ultimo_file:
        print("Errore: Nessun file di fatti trovato nella cartella 'kb/'.")
        print("Esegui prima 'python main.py' per generare il campione e la KB.")
        exit(1)

    print(f"Caricamento fatti dall'istanza registrata in: {ultimo_file}")
    img_8x8, ground_truth, target_predetto = carica_fatti_da_file(ultimo_file)
    
    print(f"Ground Truth reale: {ground_truth} | Predizione Albero registrata: {target_predetto}")
    
    caratteristiche = estrai_caratteristiche_topologiche(img_8x8)
    valida_con_ontologia(caratteristiche, target_predetto, cartella="kb")