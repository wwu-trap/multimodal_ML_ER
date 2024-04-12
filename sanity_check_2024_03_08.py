# -------------------- GENERATED WITH PHOTON WIZARD (beta) ------------------------------
# PHOTON Project Folder: C:/Users/j_boeh06/Sciebo/Forschung/PythonProjects/multimodal_ER


import pandas as pd
import numpy as np
from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch, Preprocessing
from photonai.optimization import Categorical, IntegerRange, FloatRange
from sklearn.model_selection import KFold

# Specify how results are going to be saved
# Define hyperpipe
hyperpipe = Hyperpipe('sanity_check_kevin',
                      project_folder="C:/Users/j_boeh06/Sciebo/Forschung/PythonProjects/multimodal_ER/sanity_check",
                      optimizer="grid_search",
                      optimizer_params={},
                      metrics=['mean_squared_error', 'mean_absolute_error', 'explained_variance', 'pearson_correlation',
                               'r2'],
                      best_config_metric="mean_squared_error",
                      outer_cv=KFold(n_splits=10, shuffle=True, random_state=42),
                      inner_cv=KFold(n_splits=10, shuffle=True, random_state=42),
                      verbosity=1)




# Add preprocessing: Robust Scaler
hyperpipe += PipelineElement("RobustScaler", hyperparameters={},
                             test_disabled=False, with_centering=True, with_scaling=True)

# Add feature engineering via Switch
# Tests three options PCA, off (when PCA disabled = true) and FSelect
transformer_switch = Switch('TransformerSwitch')
transformer_switch += PipelineElement("PCA", hyperparameters={'n_components':None},
                             test_disabled=True)
transformer_switch += PipelineElement("FRegressionSelectPercentile", hyperparameters={'percentile': [5, 10, 50]},
                             test_disabled=False)
hyperpipe += transformer_switch

# Add estimator
# Defaults as defined in the model zoo default parameters: https://github.com/wwu-mmll/photonai/blob/main/photonai/base/model_zoo.py
estimator_switch = Switch('EstimatorSwitch')
# I could save some time here by removing 1e-8 (0.00000001) and 1e8 (100,000,000) without losing too much
estimator_switch += PipelineElement("SVR", hyperparameters={'C': [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8], 'kernel': ['linear', 'rbf']},
                                    max_iter=1000)
estimator_switch += PipelineElement("RandomForestRegressor", hyperparameters={'max_features': ['sqrt', 'log2'],
                                                                              'min_samples_leaf': [0.01, 0.1, 0.2]})
hyperpipe += estimator_switch

# Load data
# Adjust this to the modality and its data sheet!
df = pd.read_excel('C:/Users/j_boeh06/Sciebo/Forschung/PythonProjects/multimodal_ER/sanity_check/d_feat_sanity_with_labels.xlsx')
X = np.asarray(df.iloc[:, 1:100])
y = np.asarray(df.iloc[:, 0])

# Fit hyperpipe
hyperpipe.fit(X, y)

# Feature Importances
r = hyperpipe.get_permutation_feature_importances(n_repeats=50, random_state=0)
print(r)

for i in r["mean"].argsort()[::-1]:
    if r["mean"][i] - 2 * r["std"][i] > 0:
#Hier Anpassungen von mir: Bezug auf df, Schreiben der entsprchenden Spalten-Header vor die Durchschnitts- und SD-Werte
        print(f"{df.columns[i]:<8}: "
              f"{r['mean'][i]:.3f}"
              f" +/- {r['std'][i]:.3f}")

# Schreiben der Feature Importances in eine CSV-Datei
feature_importance = pd.DataFrame(r)
feature_importance.to_csv('feature_importances.csv')