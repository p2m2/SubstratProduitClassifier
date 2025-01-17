import pandas as pd
import numpy as np
from numpy import median
from sklearn.preprocessing import StandardScaler
from statsmodels.api import add_constant, OLS
from statsmodels.stats import multitest
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import box
from jinja2 import Environment, FileSystemLoader
import os
from datetime import datetime
from io import StringIO


# Initialiser la console Rich
console = Console()

def load_data(file_path):
    sample_metadata = pd.read_excel(file_path, sheet_name='sample_metadata')
    auc_data = pd.read_excel(file_path, sheet_name='AUC_data')
    return sample_metadata, auc_data

def preprocess_data(sample_metadata, auc_data):
    scaler = StandardScaler()
    
    #columns_to_scale = ['injectionOrder', 'vol_O2', 'num_prelevement', 'cumul_O2']
    #sample_metadata[columns_to_scale] = scaler.fit_transform(sample_metadata[columns_to_scale])
    
    X = sample_metadata['cumul_O2']  # This is now the explanatory variable
    
    y_all = auc_data.drop(columns=['name'])
    y_all = scaler.fit_transform(y_all)  # This is now the response variable
    
    return X, y_all, sample_metadata


def create_model_data(X_feature, y, sample_metadata, fixed_effects, covariates):
    data = pd.DataFrame({
        'x': X_feature,
        'y': y,
    })
    
    for effect in fixed_effects:
        data[effect] = sample_metadata[effect]
    
    data = pd.get_dummies(data, columns=fixed_effects, drop_first=True)
    
    for covariate in covariates:
        data[covariate] = sample_metadata[covariate]
    
    val_dummies_fix = data.filter(regex='^(' + '|'.join(fixed_effects) + '_)').columns.tolist()
    data[val_dummies_fix] = data[val_dummies_fix].astype(int)
    
    return data, val_dummies_fix + covariates


def fit_models(data, y, covariates):
    X = add_constant(data[['x'] + covariates])
    model_lin = OLS(y, X).fit()
    
    data['x_squared'] = data['x'] ** 2
    X_quad = add_constant(data[['x', 'x_squared'] + covariates])
    model_quad = OLS(y, X_quad).fit()
    
    data['x_cubed'] = data['x'] ** 3
    X_cubic = add_constant(data[['x', 'x_squared', 'x_cubed'] + covariates])
    model_cubic = OLS(y, X_cubic).fit()
    
    return model_lin, model_quad, model_cubic

def analyze_feature(X_feature, y, sample_metadata, feature_name, fixed_effects, covariates):
    data, model_covariates = create_model_data(X_feature, y, sample_metadata, fixed_effects, covariates)
    model_lin, model_quad, model_cubic = fit_models(data, y, model_covariates)
    
    return {
        'Feature': feature_name,
        'P-value_linear': model_lin.pvalues.get('x', None),
        'Pente_linear': model_lin.params['x'],
        'R2_linear': model_lin.rsquared,
        'F-statistic_linear': model_lin.fvalue,
        'Confidence_interval_linear': model_lin.conf_int().loc['x'].tolist(),
        'P-value_quadratic': model_quad.pvalues.get('x_squared', None),
        'Pente_quadratic': model_quad.params['x_squared'],
        'R2_quadratic': model_quad.rsquared,
        'F-statistic_quadratic': model_quad.fvalue,
        'Confidence_interval_quadratic': model_quad.conf_int().loc['x_squared'].tolist(),
        'P-value_cubic': model_cubic.pvalues.get('x_cubed', None),
        'Pente_cubic': model_cubic.params['x_cubed'],
        'R2_cubic': model_cubic.rsquared,
        'F-statistic_cubic': model_cubic.fvalue,
    }


def get_type_feature(res):
    types = []
    if res['P-value_linear'] < 0.05:
        types.append('Produit' if res['Pente_linear'] > 0 else 'Substrat')
    if res['P-value_quadratic'] < 0.05 and res['Pente_quadratic'] > 0 :
        types.append('Transitoire')
    if res['P-value_cubic'] < 0.05 and res['Pente_cubic'] > 0:
        types.append('Recyclé')
    return ','.join(types) if types else 'Non classé'

def analyze_data(X, y_all, sample_metadata, feature_names, fixed_effects, covariates):
    results = []
    for index in track(range(y_all.shape[0]), description="Analysing features"):
        result = analyze_feature(X, y_all[index], sample_metadata, feature_names[index], fixed_effects, covariates)
        results.append(result)
    
    if False:
        fdr = multitest.fdrcorrection([x['P-value_linear'] for x in results], method='indep')[1]
        for i in range(len(results)):
            results[i]['P-value_linear']=fdr[i]
        
        fdr = multitest.fdrcorrection([x['P-value_quadratic'] for x in results], method='indep')[1]
        for i in range(len(results)):
            results[i]['P-value_quadratic']=fdr[i]
        fdr = multitest.fdrcorrection([x['P-value_cubic'] for x in results], method='indep')[1]
        for i in range(len(results)):
            results[i]['P-value_cubic']=fdr[i]
            
    for i in range(len(results)):
        results[i]['Type'] = get_type_feature(results[i]) 
    
    return results


def get_features_csv_name(model):
    suff = '_'.join(model['fixed_effects']) + '_'.join(model['covariates'])
    return f'Features_Results_Model_{suff}.csv'

def save_results(results, model):
    pd.DataFrame(results).to_csv(get_features_csv_name(model), index=False, float_format="%.5f")
    
    
def categorize_results(results):
    categories = {
        'produits': [x['Feature'] for x in results if 'Produit' in x['Type']],
        'substrats': [x['Feature'] for x in results if 'Substrat' in x['Type']],
        'transitoires': [x['Feature'] for x in results if 'Transitoire' in x['Type']],
        'recycles': [x['Feature'] for x in results if 'Recyclé' in x['Type']],
        'non_classes': [x['Feature'] for x in results if x['Type'] == 'Non classé'],
        'Total': [x['Feature'] for x in results]
    }
    exclusive_categories = {
        'produits_exc': [x['Feature'] for x in results if x['Type'] == 'Produit'],
        'substrats_exc': [x['Feature'] for x in results if x['Type'] == 'Substrat'],
        'transitoires_exc': [x['Feature'] for x in results if x['Type'] == 'Transitoire'],
        'recycles_exc': [x['Feature'] for x in results if x['Type'] == 'Recyclé'],
        'non_classes': [x['Feature'] for x in results if x['Type'] == 'Non classé'],
        'Total': [x['Feature'] for x in results if x['Type'].count(',') == 0]
    }
    return categories, exclusive_categories

def create_summary_table(categories, exclusive_categories):
    table = Table(title="Résumé des Résultats")
    table.add_column("Catégorie", style="cyan")
    table.add_column("Éligible", justify="right")
    table.add_column("Exclusif (1 seule catégorie)", justify="right")
    count_cat = {}
    
    for cat, exc_cat in zip(categories.items(), exclusive_categories.items()):
        table.add_row(cat[0].capitalize(), str(len(cat[1])), str(len(exc_cat[1])))
        count_cat[cat[0].capitalize()] = len(cat[1])
    
    return table,count_cat

def create_2d_table(categories, exclusive_categories):
    cat_names = ["Produit", "Substrat", "Transitoire", "Recyclé"]
    cat_keys = ["produits", "substrats", "transitoires", "recycles"]  # Notez "recycles" au lieu de "recyclés"
    table_2d = np.zeros((4, 4), dtype=int)
    
    for i, (cat1, key1) in enumerate(zip(cat_names, cat_keys)):
        for j, (cat2, key2) in enumerate(zip(cat_names, cat_keys)):
            if i == j:
                table_2d[i][j] = len(exclusive_categories[f"{key1}_exc"])
            else:
                set1 = set(categories[key1])
                set2 = set(categories[key2])
                table_2d[i][j] = len(set1.intersection(set2))
    
    table = Table(title="Tableau 2D des catégories")
    table.add_column("Catégorie", style="cyan")
    for cat in cat_names:
        table.add_column(cat, justify="right")
    
    for i, cat in enumerate(cat_names):
        table.add_row(cat, *[str(x) for x in table_2d[i]])
    
    return table

def create_slopes_table(results, categories):
    cat_names = ["Produit", "Substrat", "Transitoire", "Recyclé"]
    cat_keys = ["produits", "substrats", "transitoires", "recycles"]
    table_slopes = np.zeros((6, 4), dtype=int)
    
    for i, (category, key) in enumerate(zip(cat_names, cat_keys)):
        category_list = categories[key]
        
        # Pentes linéaires
        positive_lin = sum(1 for x in results if x['Feature'] in category_list and x['Pente_linear'] > 0 and x['P-value_linear'] < 0.05)
        negative_lin = sum(1 for x in results if x['Feature'] in category_list and x['Pente_linear'] < 0 and x['P-value_linear'] < 0.05)
        
        # Pentes quadratiques
        positive_quad = sum(1 for x in results if x['Feature'] in category_list and x['Pente_quadratic'] > 0 and x['P-value_quadratic'] < 0.05)
        negative_quad = sum(1 for x in results if x['Feature'] in category_list and x['Pente_quadratic'] < 0 and x['P-value_quadratic'] < 0.05)
        
        # Pentes cubiques
        positive_cub = sum(1 for x in results if x['Feature'] in category_list and x['Pente_cubic'] > 0 and x['P-value_cubic'] < 0.05)
        negative_cub = sum(1 for x in results if x['Feature'] in category_list and x['Pente_cubic'] < 0 and x['P-value_cubic'] < 0.05)
        
        table_slopes[:, i] = [positive_lin, negative_lin, positive_quad, negative_quad, positive_cub, negative_cub]
    
    table = Table(title="Tableau des pentes (avec p-value<0.05 associée) par catégorie")
    table.add_column("Pente", style="cyan")
    for cat in cat_names:
        table.add_column(cat, justify="right")
    
    table.add_row("Linéaire Positive", *[str(x) for x in table_slopes[0]])
    table.add_row("Linéaire Négative", *[str(x) for x in table_slopes[1]])
    table.add_row("Quadratique Positive", *[str(x) for x in table_slopes[2]])
    table.add_row("Quadratique Négative", *[str(x) for x in table_slopes[3]])
    table.add_row("Cubique Positive", *[str(x) for x in table_slopes[4]])
    table.add_row("Cubique Négative", *[str(x) for x in table_slopes[5]])
    
    return table

def analyze_slopes(results,return_data=False):
    slopes = {
        'linear': [r['Pente_linear'] for r in results],
        'quadratic': [r['Pente_quadratic'] for r in results],
        'cubic': [r['Pente_cubic'] for r in results]
    }
    
    table = Table(title="Distribution des valeurs de pente", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Statistique", style="cyan")
    table.add_column("Linéaire", justify="right")
    table.add_column("Quadratique", justify="right")
    table.add_column("Cubique", justify="right")
    
    for stat in ['Min', 'Max', 'Moyenne', 'Médiane']:
        row = [stat]
        for slope_type in ['linear', 'quadratic', 'cubic']:
            if stat == 'Min':
                value = min(slopes[slope_type])
            elif stat == 'Max':
                value = max(slopes[slope_type])
            elif stat == 'Moyenne':
                value = np.mean(slopes[slope_type])
            else:  # Médiane
                value = np.median(slopes[slope_type])
            row.append(f"{value:.4f}")
        table.add_row(*row)
    
    if return_data:
        return {
            'linear': slopes['linear'],
            'quadratic': slopes['quadratic'],
            'cubic': slopes['cubic']
        }
    
    console.print(table)

def analyze_model_quality(results, return_data=False):
    r_squared = {
        'linear': [r['R2_linear'] for r in results],
        'quadratic': [r['R2_quadratic'] for r in results],
        'cubic': [r['R2_cubic'] for r in results]
    }
    f_statistic = {
        'linear': [r['F-statistic_linear'] for r in results],
        'quadratic': [r['F-statistic_quadratic'] for r in results],
        'cubic': [r['F-statistic_cubic'] for r in results]
    }
    p_value = {
        'linear': [r['P-value_linear'] for r in results],
        'quadratic': [r['P-value_quadratic'] for r in results],
        'cubic': [r['P-value_cubic'] for r in results]
    }
    
    table = Table(title="Qualité des modèles", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Métrique", style="cyan")
    table.add_column("Linéaire", justify="right")
    table.add_column("Quadratique", justify="right")
    table.add_column("Cubique", justify="right")
    
    table.add_row("R² moyen", 
                  f"{np.mean(r_squared['linear']):.4f}",
                  f"{np.mean(r_squared['quadratic']):.4f}",
                  f"{np.mean(r_squared['cubic']):.4f}")
    
    table.add_row("F-statistique moyenne", 
                  f"{np.mean(f_statistic['linear']):.4f}",
                  f"{np.mean(f_statistic['quadratic']):.4f}",
                  f"{np.mean(f_statistic['cubic']):.4f}")
    
    table.add_row("P-value moyenne", 
                  f"{np.mean(p_value['linear']):.4f}",
                  f"{np.mean(p_value['quadratic']):.4f}",
                  f"{np.mean(p_value['cubic']):.4f}")
    
    if return_data:
        return {
            'r_squared': r_squared,
            'f_statistic': f_statistic,
            'p_value': p_value
        }
        
    console.print(table)

def correlation_matrix(results, feature_names,y_all,return_data=False):
    # Sélectionner les features les plus significatives (par exemple, top 10 avec le R² le plus élevé)
    top_features = sorted(results, key=lambda x: x['R2_linear'], reverse=True)[:10]
    feature_names_list = feature_names.tolist() if hasattr(feature_names, 'tolist') else feature_names
    
    # Créer un DataFrame avec les valeurs de ces features
    df = pd.DataFrame({r['Feature']: y_all[feature_names_list.index(r['Feature'],)] for r in top_features})
    
    # Calculer la matrice de corrélation
    corr_matrix = df.corr()
    
    # Afficher la matrice de corrélation
    table = Table(title="Matrice de corrélation des features les plus significatives", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Feature", style="cyan")
    for feature in corr_matrix.columns:
        table.add_column(feature, justify="right")
    
    for feature, row in corr_matrix.iterrows():
        table.add_row(feature, *[f"{val:.2f}" for val in row])
    
    if return_data:
        return corr_matrix.to_dict()
    
    console.print(table)

# Dans votre fonction main ou là où vous traitez vos résultats :
def display_global_analysis(results):    
    analyze_model_quality(results)
    console.print()
    
def generate_html_report(all_results, all_categories, all_exclusive_categories, y_all, feature_names, models):
    env = Environment(loader=FileSystemLoader('.'))
    env.filters['mean'] = lambda x: np.mean(x)
    env.filters['median'] = lambda x: np.median(x)
    env.filters['min'] = min
    env.filters['max'] = max
    template = env.get_template('report_template.html')

    def table_to_html(table):
        console = Console(file=StringIO(), force_terminal=False)
        console.print(table)
        html = console.file.getvalue()
        return f"<pre>{html}</pre>"

    model_reports = []
    for i, (results, categories, exclusive_categories, model) in enumerate(zip(all_results, all_categories, all_exclusive_categories, models)):
        summary_table,count_cat = create_summary_table(categories, exclusive_categories)
        table_2d = create_2d_table(categories, exclusive_categories)
        slopes_table = create_slopes_table(results, categories)
        slopes_data = analyze_slopes(results, return_data=True)
        model_quality_data = analyze_model_quality(results, return_data=True)
        correlation_data = correlation_matrix(results, feature_names, y_all, return_data=True)
        
        model_reports.append({
            'model_formula' : f"y ~ x + {' + '.join(model['fixed_effects'])} + {' + '.join(model['covariates'])}", 
            'CSV': get_features_csv_name(model),
            'fixed_effects': model['fixed_effects'],
            'covariates': model['covariates'],
            'summary_table': table_to_html(summary_table),
            'table_2d': table_to_html(table_2d),
            'slopes_table': table_to_html(slopes_table),
            'slopes_data': slopes_data,
            'model_quality_data': model_quality_data,
            'correlation_data': correlation_data,
            'count_cat' : count_cat,
            'results' : pd.DataFrame(results).drop(columns=[]).to_html(table_id=f"csvTable_{i}")
        })

    html_content = template.render(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        models=model_reports
    )

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("Rapport HTML généré avec succès !")

def main():
    file_path = 'data-test/data_M2PHENOX_AD_v2.xlsx'
    sample_metadata, auc_data = load_data(file_path)
    X, y_all, sample_metadata = preprocess_data(sample_metadata, auc_data)

    models = [
       {'fixed_effects': ['cultivar', 'repeat', 'batch'], 'covariates': []},
       {'fixed_effects': ['cultivar', 'repeat', 'batch', 'num_prelevement'], 'covariates': []},
       {'fixed_effects': ['cultivar', 'repeat', 'batch'], 'covariates': ['injectionOrder']},
      # {'fixed_effects': ['cultivar', 'repeat', 'batch'], 'covariates': ['niveau_o2']},
       {'fixed_effects': ['cultivar', 'repeat', 'batch', 'num_prelevement'], 'covariates': ['injectionOrder']},
      # {'fixed_effects': ['cultivar', 'repeat', 'batch', 'num_prelevement'], 'covariates': ['niveau_o2']},
      # {'fixed_effects': ['cultivar', 'repeat', 'batch'], 'covariates': ['injectionOrder', 'niveau_o2']},
    # {'fixed_effects': ['cultivar', 'repeat', 'batch', 'num_prelevement'], 'covariates': ['injectionOrder', 'niveau_o2']},
    ]

    all_results = []
    all_categories = []
    all_exclusive_categories = []

    for model_index, model in enumerate(models, 1):
        results = analyze_data(X, y_all, sample_metadata, auc_data['name'], model['fixed_effects'], model['covariates'])
        categories, exclusive_categories = categorize_results(results)
        if len(results) == 0:
            console.print(f"Aucune feature significative trouvée pour le modèle {model_index}")
            continue

        console.print(f"[bold]Model[/bold] : {model}\n")
        save_results(results, model)
        console.print(create_summary_table(categories, exclusive_categories))
        console.print(create_2d_table(categories, exclusive_categories))
        #console.print(create_slopes_table(results, categories))
        display_global_analysis(results)
        
        all_results.append(results)
        all_categories.append(categories)
        all_exclusive_categories.append(exclusive_categories)

    generate_html_report(all_results, all_categories, all_exclusive_categories, y_all, auc_data['name'], models)

if __name__ == "__main__":
    main()

