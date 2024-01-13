import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import warnings
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#ignore warning message
warnings.filterwarnings(action='ignore')



def data_load(target1, target2):

    # Setting data dir
    data_dir= 'data/EEG_ML_data_240111.csv'
    rsn_dir = 'data/RSN_240111.csv'
    dmn_dir = 'data/DMN_240111.csv'

    df = pd.read_csv(data_dir,  index_col='Unnamed: 0')
    
    # missing data replacement
    df = df.replace(999, np.nan)
    idx_no_channel = df[df['Ab_FP1_D'].isnull()].index
    df = df.drop(idx_no_channel)
    df = df.where(pd.notnull(df), df.mean(), axis= 1)
    df = df[df['IQ'] >= 80]
    
    # target group
    df = df[(df['target'] == target1) | (df['target'] == target2)]
    
    # Absolute & relative power & coherence extraction
    col_name = []
    for idx in df.columns.to_list():
        if 'Ab_' in idx:
            col_name.append(idx)
        if 'Rel_' in idx:
            col_name.append(idx)
        if 'Coh' in idx:
            col_name.append(idx)
            
    # DMN & RSN merging
    salireward = pd.read_csv(rsn_dir)
    RSNlist = salireward['node'].to_list()
    dmn = pd.read_csv(dmn_dir)
    dmnlist = dmn['node'].to_list()
    DMNRSN = dmnlist + RSNlist
    DMNRSN = list(set(DMNRSN))

    pie = []
    for band in ['_D', '_T','_A', '_B', '_G']:
        for i in DMNRSN:
            pie.append(i+band)
    
    col_name = col_name + pie

    col_name = col_name + ['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total', 'target']
    df_part = df[col_name]
    df_part.reset_index(drop=True, inplace = True)
        
    
    return df_part


# GridSearch & modelling 
def RF_model(X, y):
   
    n_divisions= 100
    clinic_coef_arr={}
    EEG_coef_arr={}
    best_params=[]
    
    for idx in tqdm(range(n_divisions)):
        clinic_coef, clinic_col_list, EEG_coef, EEG_col_list, params_=cal_perf(X,y,idx)

        for i_feature, feature_name in enumerate(clinic_col_list):
            if feature_name in clinic_coef_arr.keys():
                clinic_coef_arr[feature_name].append(clinic_coef[i_feature])
            else:
                clinic_coef_arr[feature_name]=[clinic_coef[i_feature]]
        
        for i_feature, feature_name in enumerate(EEG_col_list):
            if feature_name in EEG_coef_arr.keys():
                EEG_coef_arr[feature_name].append(EEG_coef[i_feature])
            else:
                EEG_coef_arr[feature_name]=[EEG_coef[i_feature]]            
        best_params=best_params+[params_]

    return clinic_coef_arr, EEG_coef_arr, best_params


def cal_perf(X,y,i_division):
    rf = RandomForestClassifier()
    scoring={'Deviance':'neg_log_loss'}
    grid={'n_estimators': [500,1000,1500,2000],'max_depth':[4,6,8,10],
         'min_samples_split':[5,10,50]}
    grid_cv = GridSearchCV(rf, param_grid=grid, 
                           cv=StratifiedKFold(n_splits=10,shuffle=True,
                                              random_state=i_division), 
                           n_jobs=40, scoring=scoring,refit='Deviance')

    #standardize X_train
    Sex=X[['Sex']]
    continuous_X=X.drop(['Sex'],axis=1)
    if not continuous_X.empty:
        scaler=StandardScaler().fit(continuous_X)
        X_train_scaled=scaler.transform(continuous_X)
        X_train_scaled=pd.DataFrame(X_train_scaled,columns=continuous_X.columns,index=Sex.index)
    else: 
        X_train_scaled=continuous_X.copy()
    X_train_scaled['Sex']=Sex


    #get clinic coef
    X_clinic_train=X_train_scaled[['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total']]
    grid_cv.fit(X_clinic_train,y)
    clinic_coef=grid_cv.best_estimator_.feature_importances_
    clinic_col_list=X_clinic_train.columns.tolist()

    clinic_i_division_train_acc=accuracy_score(y,grid_cv.predict(X_clinic_train))
    clinic_score=grid_cv.best_score_

    #get EEG coef
    X_EEG_train=X_train_scaled.drop(labels=['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total'],axis=1)
    grid_cv.fit(X_EEG_train,y)
    EEG_coef=grid_cv.best_estimator_.feature_importances_
    EEG_col_list=X_EEG_train.columns.tolist()


    EEG_score=grid_cv.best_score_

    #EEG & clinic weight
    clinic_weight=EEG_score/(clinic_score+EEG_score) # deviance is negative, so it should be inversed
    EEG_weight=1-clinic_weight

    grid_cv.best_params_['EEG_weight']=EEG_weight       

    return clinic_coef, clinic_col_list, EEG_coef, EEG_col_list, grid_cv.best_params_