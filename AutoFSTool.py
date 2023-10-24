import numpy as np
import pandas as pd 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier



def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    data = pd.read_csv(dataset_path)
    
    numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    
    player_df = data[numcols+catcols]
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
    features = traindf.columns

    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf,columns=features)
    
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']
    num_feats = 30
    # Your code ends here
    return X, y, num_feats

def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    cor_list = []
    feature_name = X.columns.tolist()
    
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    
    cor_support = [True if i in cor_feature else False for i in feature_name]
    # Your code ends here
    return cor_support, cor_feature

def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    chi2_selector = SelectKBest(score_func=chi2, k=num_feats)
    X_new = chi2_selector.fit_transform(X, y)
    
    chi_support = chi2_selector.get_support()
    chi_feature =  X.columns[chi_support] if isinstance(X, pd.DataFrame) else None
    
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    estimator = LogisticRegression(max_iter=10000000)
    rfe_selector = RFE(estimator, n_features_to_select=num_feats, step=1, verbose=5)
    
    X_new = rfe_selector.fit_transform(X, y)
    
    rfe_support = rfe_selector.support_
    rfe_feature = X.columns[rfe_support] if isinstance(X, pd.DataFrame) else None

    # Your code ends here
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)

    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", max_iter=50000), max_features=num_feats)
    
    embedded_lr_selector = embedded_lr_selector.fit(X, y)
    
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()

    # Your code ends here
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    rf = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=5), max_features=num_feats)
    rf = rf.fit(X, y)
    
    embedded_rf_support = rf.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()

    # Your code ends here
    return embedded_rf_support, embedded_rf_feature

def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    
    #### Your Code ends here
    return best_features

best_features = autoFeatureSelector(dataset_path="fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features