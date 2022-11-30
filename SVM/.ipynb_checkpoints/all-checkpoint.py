from svm_model import *
import pickle

print('HC AUD training')
df = data_load("HC", "AUD")
X = df[df.columns.difference(['target'])]
y = df[['target']]
y = y.values.ravel()
ACC_arr_test, clinic_ACC_arr_train, clinic_ACC_arr_test, EEG_ACC_arr_train, EEG_ACC_arr_test = SVM_model(X, y)
np.save('HC_AUD_feature_EEG_acc_train',EEG_ACC_arr_train)
np.save('HC_AUD_feature_EEG_acc_test',EEG_ACC_arr_test)
np.save('HC_AUD_feature_Multimodal_acc_test',ACC_arr_test)
np.save('HC_AUD_feature_NF_acc_train',clinic_ACC_arr_train)
np.save('HC_AUD_feature_NF_acc_test',clinic_ACC_arr_test)

    
    
print('HC IGD training')
df = data_load("HC", "IGD")
X = df[df.columns.difference(['target'])]
y = df[['target']]
y = y.values.ravel()
ACC_arr_test, clinic_ACC_arr_train, clinic_ACC_arr_test, EEG_ACC_arr_train, EEG_ACC_arr_test = SVM_model(X, y)
np.save('HC_IGD_feature_EEG_acc_train',EEG_ACC_arr_train)
np.save('HC_IGD_feature_EEG_acc_test',EEG_ACC_arr_test)
np.save('HC_IGD_feature_Multimodal_acc_test',ACC_arr_test)
np.save('HC_IGD_feature_NF_acc_train',clinic_ACC_arr_train)
np.save('HC_IGD_feature_NF_acc_test',clinic_ACC_arr_test)

    
    
print('AUD IDG training')
df = data_load("AUD", "IGD")
X = df[df.columns.difference(['target'])]
y = df[['target']]
y = y.values.ravel()
ACC_arr_test, clinic_ACC_arr_train, clinic_ACC_arr_test, EEG_ACC_arr_train, EEG_ACC_arr_test = SVM_model(X, y,i)
np.save('AUD_IGD_feature_EEG_acc_train',EEG_ACC_arr_train)
np.save('AUD_IGD_feature_EEG_acc_test',EEG_ACC_arr_test)
np.save('AUD_IGD_feature_Multimodal_acc_test',ACC_arr_test)
np.save('AUD_IGD_feature_NF_acc_train',clinic_ACC_arr_train)
np.save('AUD_IGD_feature_NF_acc_test',clinic_ACC_arr_test)