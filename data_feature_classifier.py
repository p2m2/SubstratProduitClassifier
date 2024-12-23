import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tabulate import tabulate
from colorama import Fore, Style,init

# Initialiser Colorama
init(autoreset=True)

# Charger le fichier Excel
file_path = 'data-test/data_M2PHENOX_AD_v2.xlsx'

# Charger les données
sample_metadata = pd.read_excel(file_path, sheet_name='sample_metadata')
auc_data = pd.read_excel(file_path, sheet_name='AUC_data')

# Extraire les données nécessaires

features_names = auc_data['name']
cultivar_Y = sample_metadata['cultivar']
repetition_Y = sample_metadata['repeat']
unique_cultivars = cultivar_Y.unique()
unique_repetitions = repetition_Y.unique()

scaler = StandardScaler()

sample_metadata[['injectionOrder', 'vol_O2', 'num_prelevement','cumul_O2']] = scaler.fit_transform(
    sample_metadata[['injectionOrder', 'vol_O2', 'num_prelevement','cumul_O2']]
)

y = sample_metadata['cumul_O2']

# Préparer les échantillons (X)
samples = auc_data.drop(columns=['name'])

samples_normalized = scaler.fit_transform(samples)

# Dictionnaire pour stocker les résultats par feature
results = []

# Modèle de régression linéaire avec pente positive
for index in tqdm(range(samples_normalized.shape[0]), desc="Analyse des lignes"):
    # Extraire les données pour la feature actuelle
    X_feature = samples_normalized[index]
    
    # Créer le DataFrame avec les données nécessaires
    data = pd.DataFrame(
        {
            'x': X_feature,
            'y': sample_metadata['cumul_O2'],
            # covariate
            'injectionOrder' : sample_metadata['injectionOrder'],
            #'vol_O2' : sample_metadata['vol_O2'],
            #'num_prelevement' : sample_metadata['num_prelevement'],
            # effet fixes
            'batch' : sample_metadata['batch'],
            'cultivar': [c.strip() for c in sample_metadata['cultivar']],
            'repeat': sample_metadata['repeat'],
            #'kinetic': sample_metadata['kinetic']
        }
    )
    
    y = data['y']
    
    # Ajouter des variables indicatrices (dummies)
    data = pd.get_dummies(data, columns=['cultivar', 'repeat', 'batch'], drop_first=True)
    
    # Utiliser filter pour obtenir les colonnes contenant les motifs souhaités
    val_dummies_fix = data.filter(regex='^(cultivar_|repeat_|batch_)').columns.tolist()
    
    data[val_dummies_fix] = data[val_dummies_fix].astype(int)
    
    # Inclure injectionOrder comme covariable dans tous les modèles
    # vol_O2 et num_prelevement sont des covariables potentielles => 
    covariates = ['injectionOrder' ] + val_dummies_fix
    
    # Modèle linéaire avec pente positive
    X = data[['x'] + covariates]
    X = sm.add_constant(X)

    # Vérifier les valeurs manquantes avant d'ajuster le modèle
    if X.isnull().any().any() or y.isnull().any():
        print(f"Valeurs manquantes détectées à l'index {index}.")
        continue  # Passer à l'itération suivante si des valeurs manquantes sont trouvées

    model_lin = sm.OLS(y, X).fit()
    pvalue_lin = model_lin.pvalues.get('x', None)
    slope_linear = model_lin.params['x']
    # Coefficient de détermination
    r_squared = model_lin.rsquared
    # Statistique F
    f_statistic = model_lin.fvalue
    # Intervalle de confiance
    conf_int = model_lin.conf_int().loc['x'].tolist()
    
    # Ajouter le terme quadratique (x^2)
    data['x_squared'] = data['x'] ** 2
    X_quad = data[['x', 'x_squared'] + covariates]
    X_quad = sm.add_constant(X_quad)
    
    model_quad = sm.OLS(y, X_quad).fit()
    pvalue_quad = model_quad.pvalues.get('x_squared', None)
    slope_quad = model_quad.params['x_squared']
    r_squared_quad = model_quad.rsquared
    f_statistic_quad = model_quad.fvalue
    conf_int_quad = model_quad.conf_int().loc['x_squared'].tolist()
    
    # Ajouter le terme cubique (x^3)
    data['x_cubed'] = data['x'] ** 3
    X_cubic = data[['x', 'x_squared', 'x_cubed'] + covariates]
    X_cubic = sm.add_constant(X_cubic)
    
    model_cubic = sm.OLS(y, X_cubic).fit()
    pvalue_cubic = model_cubic.pvalues.get('x_cubed', None)
    slope_cubic = model_cubic.params['x_cubed']
    r_squared_cubic = model_cubic.rsquared
    f_statistic_cubic = model_cubic.fvalue
    conf_int_cubic = model_cubic.conf_int().loc['x_cubed'].tolist()
     
    # Stocker les résultats pour la feature actuelle
    results.append({
        'Feature': features_names[index],
        'P-value_linear': pvalue_lin,
        'Pente_linear': slope_linear,
        'R2_linear': r_squared,
        'F-statistic_linear': f_statistic,
        'Confidence_interval_linear': conf_int,
        'P-value_quadratic': pvalue_quad,
        'Pente_quadratic': slope_quad,
        'R2_quadratic': r_squared_quad,
        'F-statistic_quadratic': f_statistic_quad,
        'Confidence_interval_quadratic': conf_int_quad,
        'P-value_cubic': pvalue_cubic,
        'Pente_cubic': slope_cubic,
        'R2_cubic': r_squared_cubic,
        'F-statistic_cubic': f_statistic_cubic,
    })

def get_type_feature(res):
    out =""
    if res['P-value_linear'] < 0.05 and res['Pente_linear']>0:
        out += 'Produit,'
    if res['P-value_linear'] < 0.05 and res['Pente_linear']<0:
        out += 'Substrat,'
    if res['P-value_quadratic'] < 0.05:
        out += 'Transitoire,'
    if res['P-value_cubic'] < 0.05:
        out += 'Recyclé,'
    
    return out[:-1]

# Sauvegarder dans un fichier CSV avec formatage des p-values
results = [{**x, 'Type': get_type_feature(x)} for x in results]
pd.DataFrame(results).to_csv('Features_Results.csv', index=False, float_format="%.5f")
pd.DataFrame({'Res': [x['Type'] for x in results]}).to_csv('Features_Results_Type.csv', index=False)

# Filtrer les résultats par catégorie
produits = [x['Feature'] for x in results if x['P-value_linear'] < 0.05 and x['Pente_linear'] > 0]
produits_exc = [x['Feature'] for x in results if x['P-value_linear'] < 0.05 and x['Pente_linear'] > 0 and x['P-value_quadratic'] >= 0.05 and x['P-value_cubic'] >= 0.05]
substrats = [x['Feature'] for x in results if x['P-value_linear'] < 0.05 and x['Pente_linear'] < 0]
substrats_exc = [x['Feature'] for x in results if x['P-value_linear'] < 0.05 and x['Pente_linear'] < 0 and x['P-value_quadratic'] >= 0.05 and x['P-value_cubic'] >= 0.05]
transitoires = [x['Feature'] for x in results if x['P-value_quadratic'] < 0.05]
transitoires_exc = [x['Feature'] for x in results if x['P-value_quadratic'] < 0.05 and x['P-value_linear'] >= 0.05 and x['P-value_cubic'] >= 0.05]
recycles = [x['Feature'] for x in results if x['P-value_cubic'] < 0.05]
recycles_exc = [x['Feature'] for x in results if x['P-value_cubic'] < 0.05 and x['P-value_linear'] >= 0.05 and x['P-value_quadratic'] >= 0.05]
non_classes = [x['Feature'] for x in results if x['P-value_cubic'] >= 0.05 and x['P-value_quadratic'] >= 0.05 and x['P-value_linear'] >= 0.05]

# Créer un tableau des résultats
table_data = [
    [f"{Fore.GREEN}Produits{Style.RESET_ALL}", len(produits), len(produits_exc)],
    [f"{Fore.RED}Substrats{Style.RESET_ALL}", len(substrats), len(substrats_exc)],
    [f"{Fore.YELLOW}Transitoires{Style.RESET_ALL}", len(transitoires),len(transitoires_exc)],
    [f"{Fore.BLUE}Recyclés{Style.RESET_ALL}", len(recycles), len(recycles_exc)],
    [f"{Fore.WHITE}Non classés{Style.RESET_ALL}", len(non_classes),len(results)-len(produits_exc)-len(substrats_exc)-len(transitoires_exc)-len(recycles_exc)],
]

# Afficher le tableau
print(f"\n{Style.BRIGHT}{Fore.CYAN}Résumé des Résultats :\n")
print(tabulate(table_data, headers=["Catégorie", "Éligible", "Exclusif (1 seule catégorie)"], tablefmt="grid"))
