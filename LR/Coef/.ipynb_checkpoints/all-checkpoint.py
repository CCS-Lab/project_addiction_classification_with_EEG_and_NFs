from LR_model import *
import pickle

print('AUD_IGD_training')
df = data_load("AUD", "IGD")
X = df[df.columns.difference(['target'])]
y = df[['target']]
y = y.values.ravel()
clinic_coef_arr, EEG_coef_arr, best_params = LR_model(X, y)
with open('AUD_IGD_feature_set_NF_coef.pickle', 'wb') as handle:
    pickle.dump(clinic_coef_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('AUD_IGD_feature_set_EEG_coef.pickle', 'wb') as handle:
    pickle.dump(EEG_coef_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('AUD_IGD_hyperparams.pickle'.format(num=i), 'wb') as handle:
    pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print('HC_AUD_training')
df = data_load("HC", "AUD")
X = df[df.columns.difference(['target'])]
y = df[['target']]
y = y.values.ravel()
clinic_coef_arr, EEG_coef_arr, best_params = LR_model(X, y)
with open('HC_AUD_feature_set_NF_coef.pickle', 'wb') as handle:
    pickle.dump(clinic_coef_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('HC_AUD_feature_set_EEG_coef.pickle', 'wb') as handle:
    pickle.dump(EEG_coef_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('HC_AUD_hyperparams.pickle'.format(num=i), 'wb') as handle:
    pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print('HC_IGD_training')
df = data_load("HC", "IGD")
X = df[df.columns.difference(['target'])]
y = df[['target']]
y = y.values.ravel()
clinic_coef_arr, EEG_coef_arr, best_params = LR_model(X, y)
with open('HC_IGD_feature_set_NF_coef.pickle', 'wb') as handle:
    pickle.dump(clinic_coef_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('HC_IGD_feature_set_EEG_coef.pickle', 'wb') as handle:
    pickle.dump(EEG_coef_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('HC_IGD_hyperparams.pickle'.format(num=i), 'wb') as handle:
    pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)