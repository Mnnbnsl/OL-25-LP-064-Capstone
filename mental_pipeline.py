import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class MentalHealthPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() 

        
        drop_cols = ['Timestamp', 'state', 'Country', 'comments']
        X = X.drop(columns=[col for col in drop_cols if col in X.columns])

        
        if 'treatment' in X.columns:
            X = X.drop(columns=['treatment'])

        
        binary_map = {'Yes': 1, 'No': 0}
        binary_cols = [
            'self_employed', 'family_history', 'remote_work', 'tech_company', 
            'benefits', 'care_options', 'wellness_program', 'seek_help', 
            'anonymity', 'mental_health_consequence', 'phys_health_consequence', 
            'mental_health_interview', 'phys_health_interview', 'obs_consequence'
        ]
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map(binary_map).fillna(0)

        
        trinary_map = {'Yes': 2, 'No': 0, "Don't know": 1, 'Some of them': 1, 'Not sure': 1, 'Maybe': 1}
        trinary_cols = ['supervisor', 'coworkers', 'mental_vs_physical']
        for col in trinary_cols:
             if col in X.columns:
                X[col] = X[col].map(trinary_map).fillna(1)
                
        
        leave_map = {
            'Very difficult': 0, 'Somewhat difficult': 1, "Don't know": 2,
            'Somewhat easy': 3, 'Very easy': 4
        }
        if 'leave' in X.columns:
            X['leave'] = X['leave'].map(leave_map).fillna(2)

        interfere_map = {
            'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3
        }
        if 'work_interfere' in X.columns:
            X['work_interfere'] = X['work_interfere'].map(interfere_map).fillna(2)

        size_map = {
            '1-5': 0, '6-25': 1, '26-100': 2,
            '100-500': 3, '500-1000': 4, 'More than 1000': 5
        }
        if 'no_employees' in X.columns:
            X['no_employees'] = X['no_employees'].map(size_map).fillna(2)

        if 'Gender' in X.columns:
            def clean_gender(gender):
                if not isinstance(gender, str): 
                    return 'Other/Non-Binary'
                g = gender.lower()
                if 'fem' in g or 'wom' in g or g == 'f':
                    return 'Female'
                elif 'mal' in g or 'man' in g or g == 'm' or 'guy' in g:
                    return 'Male'
                else:
                    return 'Other/Non-Binary'
            
            X['Gender'] = X['Gender'].apply(clean_gender)
            le = LabelEncoder()
            X['Gender'] = le.fit_transform(X['Gender'])

        return X
    

class MentalHealthRegressionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='Age'):
        self.target_col = target_col

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        X = X.copy()

        cols_to_drop = ['Timestamp', 'state', 'Country', 'comments', self.target_col]
        X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])

        binary_map = {'Yes': 1, 'No': 0}
        binary_cols = [
            'self_employed', 'family_history', 'treatment', 'remote_work', 'tech_company', 
            'benefits', 'care_options', 'wellness_program', 'seek_help', 
            'anonymity', 'mental_health_consequence', 'phys_health_consequence', 
            'mental_health_interview', 'phys_health_interview', 'obs_consequence'
        ]
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map(binary_map).fillna(0)

        trinary_map = {'Yes': 2, 'No': 0, "Don't know": 1, 'Some of them': 1, 'Not sure': 1, 'Maybe': 1}
        trinary_cols = ['supervisor', 'coworkers', 'mental_vs_physical']
        for col in trinary_cols:
             if col in X.columns:
                X[col] = X[col].map(trinary_map).fillna(1) # Fill NaNs with the neutral value (1)
                
        leave_map = {
            'Very difficult': 0, 'Somewhat difficult': 1, "Don't know": 2,
            'Somewhat easy': 3, 'Very easy': 4
        }
        if 'leave' in X.columns:
            X['leave'] = X['leave'].map(leave_map).fillna(2) # Fill NaNs with the neutral value (2)

        interfere_map = {
            'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3
        }
        if 'work_interfere' in X.columns:
            X['work_interfere'] = X['work_interfere'].map(interfere_map).fillna(2)

        size_map = {
            '1-5': 0, '6-25': 1, '26-100': 2,
            '100-500': 3, '500-1000': 4, 'More than 1000': 5
        }
        if 'no_employees' in X.columns:
            X['no_employees'] = X['no_employees'].map(size_map).fillna(2) # Fill NaNs with a medium size

    
        if 'Gender' in X.columns:
            def clean_gender(gender):
                if not isinstance(gender, str): 
                    return 'Other/Non-Binary'
                g = gender.lower().strip()
                if 'fem' in g or 'wom' in g or g == 'f':
                    return 'Female'
                elif 'mal' in g or 'man' in g or g == 'm' or 'guy' in g:
                    return 'Male'
                else:
                    return 'Other/Non-Binary'
            
            X['Gender'] = X['Gender'].apply(clean_gender)
            le = LabelEncoder()
            X['Gender'] = le.fit_transform(X['Gender'])

        return X