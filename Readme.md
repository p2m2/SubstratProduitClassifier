# Analyse des Features avec Régression Linéaire, Quadratique et Cubique

Ce programme permet d'analyser des features issues de données expérimentales en utilisant des modèles de régression linéaire, quadratique et cubique pour prédire le cumul d'oxygène (cumul_O2) en tant que variable dépendante (y). Les analyses sont effectuées en quatre paquets distincts, chacun correspondant à un type de relation entre les features et le cumul_O2 :

- Modèle de Régression Linéaire (lm1) :
    - Ce modèle est utilisé pour analyser les substrats, où l'abondance des substrats suit une tendance décroissante au fil du temps : abondance à t1 > t2 > t3.
- Modèle de Régression Linéaire (lm2) :
    - Ce modèle s'applique aux produits, où l'abondance des produits montre une tendance croissante : abondance à t1 < t2 < t3.
- Modèle de Régression Quadratique (lm3) :
    - Ce modèle est destiné aux features transitoires, représentant une relation non linéaire : abondance à t1 < t2 > t3, indiquant un pic d'abondance à t2 suivi d'une diminution.
- Modèle de Régression Cubique (lm4) :
    - Ce modèle analyse les features recyclées, où l'abondance évolue selon un modèle plus complexe : abondance à t1 < t2 > t3 < t4, montrant des variations significatives au fil du temps.


## Installation
### Prérequis

```bash
python -m venv env
source env/bin/activate
pip install pandas statsmodels scikit-learn tqdm colorama tabulate
```

## Exécution

- data-test/data_M2PHENOX_AD_v2.xlsx.
- Le fichier doit contenir deux feuilles :
    - sample_metadata : Contient les métadonnées des échantillons (par exemple, injectionOrder, vol_O2, etc.).
    - AUC_data : Contient les features à analyser (par exemple, les noms des features et leurs valeurs).

```bash
python data_feature_classifier.py
```

## Résultats

- Un résumé des résultats affiché dans la console.
- Deux fichiers CSV générés dans le répertoire courant :

    - Features_Results.csv
    - Features_Results_Type.csv

### Affichage Console

- Le programme affiche un tableau récapitulatif des catégories de features analysées :

    - Produits : Features identifiées comme ayant une pente linéaire positive significative.
    - Substrats : Features avec une pente linéaire négative significative.
    - Transitoires : Features présentant une relation quadratique significative.
    - Recyclés : Features présentant une relation cubique significative.
    - Non classés : Features ne correspondant à aucune des catégories ci-dessus.

- Chaque catégorie est divisée en deux colonnes :

    - Éligible : Nombre total de features appartenant à cette catégorie.
    - Exclusif (1 seule catégorie) : Nombre de features appartenant exclusivement à cette catégorie.

### Fichier CSV : Features_Results.csv

- Ce fichier contient les résultats détaillés pour chaque feature. Les colonnes incluent :

    - Feature : Nom de la feature.
    - P-value_linear, Pente_linear, R2_linear, etc. : Statistiques du modèle linéaire.
    - P-value_quadratic, etc. : Statistiques du modèle quadratique.
    - P-value_cubic, etc. : Statistiques du modèle cubique.
    - Type : Classification de la feature (Produit, Substrat, Transitoire, ou Recyclé).

### Fichier CSV : Features_Results_Type.csv

- Ce fichier contient uniquement les types associés à chaque feature.

### Personnalisation

#### Ajout de Covariables :

Dans le script, vous pouvez ajouter ou retirer des covariables en modifiant la liste suivante :

```python
covariates = ['injectionOrder', 'vol_O2', 'num_prelevement'] + val_dummies_fix
```

#### Modification des Critères de Classification :
La logique pour classifier les features peut être modifiée dans la fonction suivante :

```python
def get_type_feature(res):
    # Exemple de logique actuelle
    if res['P-value_linear'] < 0.05 and res['Pente_linear'] > 0:
        return 'Produit'
    # Ajoutez vos propres conditions ici
```