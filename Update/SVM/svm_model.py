import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
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

    if target2 == 'IGD':
        df['target'] = (df['target'] == 'IGD').astype(int)
    elif target2 == 'AUD':
        df['target'] = (df['target'] == 'AUD').astype(int)    
    
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
    for band in ["_D", "_T","_A", "_B", "_G"]:
        for i in DMNRSN:
            pie.append(i+band)
    
    col_name = col_name + pie

    col_name = col_name + ['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total', 'target']
    df_part = df[col_name]
    df_part.reset_index(drop=True, inplace = True)
        
    
    return df_part

# perfome permutation_test 
def permutation_test(y_true, y_pred, num_permutations=1000):
    observed_accuracy = accuracy_score(y_true, y_pred)
    count = 0

    for _ in range(num_permutations):
        # Permute labels
        permuted_labels = np.random.permutation(y_true)
        permuted_accuracy = accuracy_score(permuted_labels, y_pred)

        if permuted_accuracy >= observed_accuracy:
            count += 1

    p_value = count / num_permutations
    return p_value

def calculate_spec_sens(cm):

    TN, FP, FN, TP = cm.ravel()
    print(TN, FP, FN, TP)
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)
    
    return round(specificity,3), round(sensitivity,3)

# GridSearch & modelling 
def SVM_model(X, y):
   
    n_divisions= np.shape(X)[0]
    coef_arr={}
    
    clinic_ACC_arr_test=np.empty((n_divisions,1))
    clinic_ACC_arr_test[:]=np.NaN
    clinic_ACC_arr_train=np.empty((n_divisions,1))
    clinic_ACC_arr_train[:]=np.NaN

    EEG_ACC_arr_test=np.empty((n_divisions,1))
    EEG_ACC_arr_test[:]=np.NaN
    EEG_ACC_arr_train=np.empty((n_divisions,1))
    EEG_ACC_arr_train[:]=np.NaN

    ACC_arr_test=np.empty((n_divisions,1))
    ACC_arr_test[:]=np.NaN
    ACC_arr_train=np.empty((n_divisions,1))
    ACC_arr_train[:]=np.NaN

    all_y_true = []
    all_y_pred_clinic = []
    all_y_pred_EEG = []
    all_y_pred_combined = []
    
    #LOOCV
    for idx in tqdm(range(n_divisions)):
        clinic_i_division_train_acc, clinic_i_division_test_acc, EEG_i_division_train_acc, EEG_i_division_test_acc, i_division_test_acc, y_pred_clinic, y_pred_EEG, y_pred_combined, y_true =cal_perf(X,y,idx)

        ACC_arr_test[idx]=i_division_test_acc
        clinic_ACC_arr_train[idx]=clinic_i_division_train_acc
        clinic_ACC_arr_test[idx]=clinic_i_division_test_acc
        EEG_ACC_arr_train[idx]=EEG_i_division_train_acc
        EEG_ACC_arr_test[idx]=EEG_i_division_test_acc

        # Store predictions and actuals
        all_y_true.append(int(y_true))
        all_y_pred_clinic.append(int(y_pred_clinic))
        all_y_pred_EEG.append(int(y_pred_EEG))
        all_y_pred_combined.append(int(y_pred_combined[0]))
   
    ACC_arr_test=[np.mean(ACC_arr_test)]
    print(np.mean(clinic_ACC_arr_test))
    print(np.mean(EEG_ACC_arr_test))
    print(ACC_arr_test)

    # confusion matrix
    cm_clinic = confusion_matrix(all_y_true, all_y_pred_clinic)
    cm_EEG = confusion_matrix(all_y_true, all_y_pred_EEG)
    cm_combined = confusion_matrix(all_y_true, all_y_pred_combined)


    p_value_clinic = permutation_test(all_y_true, all_y_pred_clinic)
    p_value_EEG = permutation_test(all_y_true, all_y_pred_EEG)
    p_value_combined = permutation_test(all_y_true, all_y_pred_combined)

    # Print permutation test p-values
    print(f"P-value for EEG Model: {p_value_EEG}")
    print(f"P-value for Clinic Model: {p_value_clinic}")
    print(f"P-value for Combined Model: {p_value_combined}")

    # Print sensitivity & specificity
    print("EEG spec & sens", calculate_spec_sens(cm_EEG))
    print("clinical spec & sens", calculate_spec_sens(cm_clinic))
    print("combined spec & sens", calculate_spec_sens(cm_combined))

    return ACC_arr_test, clinic_ACC_arr_train, clinic_ACC_arr_test, EEG_ACC_arr_train, EEG_ACC_arr_test


def cal_perf(X,y,i_division):
    svm = CalibratedClassifierCV(base_estimator=LinearSVC(penalty="l1",max_iter=1e9, dual=False))
    scoring={'Deviance':'neg_log_loss'}
    grid=[{'base_estimator__C': np.logspace(-1.5,0.3,40)}]
    grid_cv = GridSearchCV(svm, param_grid=grid, 
                           cv=StratifiedKFold(n_splits=10,shuffle=True,
                                              random_state=i_division), 
                           n_jobs=40, scoring=scoring,refit='Deviance')
    
    #date split for LOOCV
    X_train=X.drop(i_division,axis=0)
    X_test=X.iloc[[i_division],:]
    y_train=np.delete(y,i_division)
    y_test=np.array(y[i_division],dtype=object)

    col_list=X.columns.tolist()

    #standardize X_train
    Sex=X_train[['Sex']]
    continuous_X=X_train.drop(['Sex'],axis=1)
    if not continuous_X.empty:
        scaler=StandardScaler().fit(continuous_X)
        X_train_scaled=scaler.transform(continuous_X)
        X_train_scaled=pd.DataFrame(X_train_scaled,columns=continuous_X.columns,index=Sex.index)
    else: 
        X_train_scaled=continuous_X.copy()
    X_train_scaled['Sex']=Sex

    #standardize X_test with mean and std of X_train
    Sex=X_test[['Sex']]
    continuous_X=X_test.drop(['Sex'],axis=1)
    if not continuous_X.empty:
        X_test_scaled=scaler.transform(continuous_X)
        X_test_scaled=pd.DataFrame(X_test_scaled,columns=continuous_X.columns,index=Sex.index)
    else: 
        X_test_scaled=continuous_X.copy()
    X_test_scaled['Sex']=Sex

    X_clinic_train=X_train_scaled[['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total']]
    grid_cv.fit(X_clinic_train,y_train)

    # NF model fitting
    clinic_i_division_train_acc=accuracy_score(y_train,grid_cv.predict(X_clinic_train))
    clinic_i_division_test_acc=y_test==grid_cv.predict(X_test_scaled[['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total']])
    clinic_proba=grid_cv.predict_proba(X_test_scaled[['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total']])
    clinic_score=grid_cv.best_score_
    y_pred_clinical = grid_cv.predict(X_test_scaled[['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total']])

    X_EEG_train=X_train_scaled.drop(labels=['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total'],axis=1)
    grid_cv.fit(X_EEG_train,y_train)

    #EEG model fitting
    EEG_i_division_train_acc=accuracy_score(y_train,grid_cv.predict(X_EEG_train))
    EEG_i_division_test_acc=y_test==grid_cv.predict(X_test_scaled.drop(labels=['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total'],axis=1))
    EEG_proba=grid_cv.predict_proba(X_test_scaled.drop(labels=['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total'],axis=1))
    EEG_score=grid_cv.best_score_
    y_pred_EEG = grid_cv.predict(X_test_scaled.drop(labels=['Age_', 'Sex', 'IQ', 'BDI', 'BAI', 'Barratt_total'], axis=1))

    #Combine NF and EEG models based on each model deviance
    clinic_weight=EEG_score/(clinic_score+EEG_score) # deviance is negative, so it should be inversed
    EEG_weight=1-clinic_weight
    total_proba=clinic_weight*clinic_proba[0]+EEG_weight*EEG_proba[0]
    predicted_ans=grid_cv.best_estimator_.classes_[np.argmax(total_proba)]
    i_division_test_acc=y_test==predicted_ans
    y_pred_combined = [predicted_ans]

    grid_cv.best_params_['EEG_weight']=EEG_weight

    return clinic_i_division_train_acc, clinic_i_division_test_acc, EEG_i_division_train_acc, EEG_i_division_test_acc, i_division_test_acc, y_pred_clinical, y_pred_EEG, y_pred_combined, y_test