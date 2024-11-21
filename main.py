import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_percentage_error
import statsmodels.api as sm
from scipy import stats

class WorkLifeBalanceAnalyzer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.metrics = {}
        
    def load_and_preprocess(self, filename):
        self.data = pd.read_csv(filename)
        
        if 'Timestamp' in self.data.columns:
            self.data = self.data.drop('Timestamp', axis=1)
        
        self._handle_categorical()
        
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        
        self._calculate_correlations()
        self._engineer_features()
        self.train_test_split()

    def _handle_categorical(self):
        age_mapping = {
            'Less than 20': 18,
            '21 to 35': 28,
            '36 to 50': 43,
            '51 or more': 60
        }
        if 'AGE' in self.data.columns:
            self.data['AGE_NUMERIC'] = self.data['AGE'].map(age_mapping)
            self.data = self.data.drop('AGE', axis=1)
        
        if 'GENDER' in self.data.columns:
            self.data['GENDER_ENCODED'] = (self.data['GENDER'] == 'Female').astype(int)
            self.data = self.data.drop('GENDER', axis=1)
        
        if 'DAILY_STRESS' in self.data.columns:
            def convert_stress(x):
                if isinstance(x, (int, float)):
                    return x
                elif isinstance(x, str):
                    return 1 if x.lower() == 'yes' else 0
                return None
            
            self.data['DAILY_STRESS'] = self.data['DAILY_STRESS'].apply(convert_stress)
    
    def _calculate_correlations(self):
        correlations = self.data.corr(method='spearman')['WORK_LIFE_BALANCE_SCORE'].sort_values(ascending=False)

    def _engineer_features(self):
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(self.data[numeric_features])
        numeric_features = list(np.array(numeric_features)[selector.get_support()])
        
        self.features = numeric_features
        self.X = self.data[numeric_features].copy()
        self.y = self.data['WORK_LIFE_BALANCE_SCORE'].copy()
        
        correlations = []
        for feature in numeric_features:
            corr, p_value = stats.spearmanr(self.X[feature], self.y)
            if p_value < 0.05:
                correlations.append((feature, corr))
    
    def train_test_split(self):
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.data.drop('WORK_LIFE_BALANCE_SCORE', axis=1),
            self.data['WORK_LIFE_BALANCE_SCORE'],
            test_size=0.2,
            random_state=self.random_state
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            random_state=self.random_state
        )
        
        ordinal_vars = ['BMI_RANGE', 'LOST_VACATION', 'SUFFICIENT_INCOME']
        outlier_params = {}
        
        for column in self.X_train.columns:
            if column in ordinal_vars:
                continue
                
            median = self.X_train[column].median()
            mad = stats.median_abs_deviation(self.X_train[column])
            
            if mad == 0:
                continue
                
            threshold = 4.0
            lower_bound = median - (threshold * mad / 0.6745)
            upper_bound = median + (threshold * mad / 0.6745)
            
            outlier_params[column] = {'lower': lower_bound, 'upper': upper_bound}
            self.X_train[column] = self.X_train[column].clip(lower_bound, upper_bound)
        
        for column, bounds in outlier_params.items():
            self.X_val[column] = self.X_val[column].clip(bounds['lower'], bounds['upper'])
            self.X_test[column] = self.X_test[column].clip(bounds['lower'], bounds['upper'])
        
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train[numeric_features])
        feature_names = list(numeric_features) + [f"{numeric_features[i]}_{numeric_features[j]}" 
                                                for i in range(len(numeric_features)) 
                                                for j in range(i+1, len(numeric_features))]
        
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X_train_poly)
        selected_features_mask = selector.get_support()
        
        correlations = []
        for i, (feature, mask) in enumerate(zip(feature_names, selected_features_mask)):
            if not mask:
                continue
            corr, p_value = stats.spearmanr(X_train_poly[:, i], self.y_train)
            if p_value < 0.05 and abs(corr) > 0.1:
                correlations.append((feature, abs(corr)))
        
        selected_features = [f[0] for f in sorted(correlations, key=lambda x: x[1], reverse=True)[:20]]
        
        X_val_poly = poly.transform(self.X_val[numeric_features])
        X_test_poly = poly.transform(self.X_test[numeric_features])
        
        self.X_train = pd.DataFrame(X_train_poly, columns=feature_names)[selected_features]
        self.X_val = pd.DataFrame(X_val_poly, columns=feature_names)[selected_features]
        self.X_test = pd.DataFrame(X_test_poly, columns=feature_names)[selected_features]
    
    def _check_multicollinearity(self, X):
        features_to_keep = []
        vif_data = []
        remaining_features = X.columns.tolist()
        
        while remaining_features:
            vif_stats = []
            for feature in remaining_features:
                other_features = [f for f in remaining_features if f != feature]
                if not other_features:
                    vif_stats.append((feature, 1.0))
                    continue
                    
                try:
                    X_others = sm.add_constant(X[other_features])
                    model = sm.OLS(X[feature], X_others).fit()
                    r2 = model.rsquared
                    vif = 1 / (1 - r2) if r2 < 1 else float('inf')
                    vif_stats.append((feature, vif))
                except:
                    vif_stats.append((feature, float('inf')))
            
            valid_vifs = [(f, v) for f, v in vif_stats if not np.isnan(v) and v != float('inf')]
            if not valid_vifs:
                break
                
            feature, vif = min(valid_vifs, key=lambda x: x[1])
            
            if vif < 5:
                features_to_keep.append(feature)
                remaining_features.remove(feature)
                vif_data.append({'Feature': feature, 'VIF': vif})
            else:
                break
        
        return features_to_keep
    
    def train_and_evaluate_models(self):
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=self.random_state)
        
        self.models = {
            'LASSO': Pipeline([
                ('standardscaler', StandardScaler()),
                ('lassocv', LassoCV(
                    alphas=np.logspace(-4, 1, 100),
                    cv=cv,
                    max_iter=10000,
                    random_state=self.random_state,
                    selection='random'
                ))
            ]),
            'Ridge': Pipeline([
                ('standardscaler', StandardScaler()),
                ('ridgecv', RidgeCV(
                    alphas=np.logspace(-4, 1, 100),
                    cv=cv,
                    scoring='neg_mean_squared_error'
                ))
            ]),
            'Random Forest': RandomForestRegressor(
                n_estimators=1000,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=4,
                min_samples_split=15,
                min_samples_leaf=8,
                subsample=0.8,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state
            )
        }

        self.metrics = {}
        self.feature_importance = {}

        for name, model in self.models.items():
            cv_r2 = cross_val_score(model, self.X_train, self.y_train, 
                                  cv=cv, scoring='r2')
            cv_rmse = np.sqrt(-cross_val_score(model, self.X_train, self.y_train, 
                                             cv=cv, scoring='neg_mean_squared_error'))
            cv_mae = -cross_val_score(model, self.X_train, self.y_train, 
                                    cv=cv, scoring='neg_mean_absolute_error')

            model.fit(self.X_train, self.y_train)

            val_pred = model.predict(self.X_val)
            val_r2 = r2_score(self.y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            val_mae = mean_absolute_error(self.y_val, val_pred)

            test_pred = model.predict(self.X_test)
            test_r2 = r2_score(self.y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_mae = mean_absolute_error(self.y_test, test_pred)

            self.metrics[name] = {
                'cv_r2_mean': cv_r2.mean(),
                'cv_r2_std': cv_r2.std(),
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'cv_mae_mean': cv_mae.mean(),
                'cv_mae_std': cv_mae.std(),
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'test_r2': test_r2
            }

            if name in ['Random Forest', 'Gradient Boosting']:
                importance = model.feature_importances_
                feature_imp = pd.DataFrame({
                    'Feature': self.X_train.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                self.feature_importance[name] = feature_imp
            elif name in ['LASSO', 'Ridge']:
                if hasattr(model[-1], 'coef_'):
                    importance = np.abs(model[-1].coef_)
                    feature_imp = pd.DataFrame({
                        'Feature': self.X_train.columns,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    self.feature_importance[name] = feature_imp

def main():
    analyzer = WorkLifeBalanceAnalyzer()
    analyzer.load_and_preprocess('dataset.csv')
    analyzer.train_and_evaluate_models()

if __name__ == "__main__":
    main()
