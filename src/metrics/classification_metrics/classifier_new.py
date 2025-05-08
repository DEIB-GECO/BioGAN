import numpy as np 
import random
import torch
import torch.nn as nn
import warnings
import matplotlib as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import auc, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from classification.utils import *
from model_utils import *
from sklearn.utils import shuffle
from metrics.correlation_score import gamma_coef, gamma_coefficients
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, precision_score,  balanced_accuracy_score, f1_score, roc_curve, auc
from metrics.precision_recall import get_precision_recall
# Set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


import warnings

warnings.filterwarnings("ignore")


def remap_labels(y):
    unique_labels = torch.unique(y)
    label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
    y_mapped = torch.tensor([label_map[label.item()] for label in y], device=y.device)
    return y_mapped, len(unique_labels)  

class Predictor():
    
    def __init__(self,model, task=None):

        self.model = model
        self.task = task
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        '''
         # Imbalance weights
        if class_weights is None:
            class_weights = torch.ones(size=(self.nb_classes,))/self.nb_classes
        else:
            binary_weights, cancer_weights , tissue_weights = class_weights
            if self.task=='binary':
                class_weights = binary_weights.to(self.device)
                cancer_weights = []
                tissue_weights = []
            elif self.task=='cancer_type':
                class_weights = cancer_weights.to(self.device)
                binary_weights = []
                tissue_weights = []
            elif self.task=='tissue_type':
                class_weights = tissue_weights.to(self.device)
                binary_weights = []
                cancer_weights = []
        '''
       

    def train_classifiers(self):
        pass
    

        
    def generate_samples(self, data):

        all_real  = []
        all_gen = []
        all_tissue = []

        for i, data in enumerate(data):
                
            x_GE = data[0].to(self.device)
            x_cat =  data[1].to(self.device)
           
            
            x_real, x_gen = self.model.generate_samples(x_GE, x_cat)
            if i==0:
                all_gen =  x_gen.cpu().detach().numpy()
                all_real = x_real.cpu().detach().numpy()
            else: 
                all_gen = np.append(all_gen,  x_gen.cpu().detach().numpy(), axis=0)
                all_real = np.append(all_real, x_real.cpu().detach().numpy(), axis=0)

            data_t = data[1].t()
            all_tissue.extend(data_t[0].numpy()) #considered order: ['tissue', 'tissue_specific', 'age','sex'] 
        
        #all_tissue,_ = remap_labels(torch.tensor(all_tissue)) #mapping for the MLP problem with labels
        

        all_real_x = np.vstack(all_real)
        all_real_gen = np.vstack(all_gen)
        
        print("all_real_x shape", len(all_real_x))
        #print("all_tissue_label", all_tissue)

        return all_real_x, all_real_gen, np.array(all_tissue)
    
    
    
    def load_real_test_data(self, data):

        all_real  = []
        all_tissue = []
        for i, data in enumerate(data):
            x_real = data[0]
            x_cat =  data[1]
            x_num = data[2]
            if i==0:
                all_real = x_real.numpy()
            else: 
                all_real = np.append(all_real, x_real.numpy(), axis=0)

            data_t = data[1].t()
            all_tissue.extend(data_t[0].numpy())
            

        all_real_x = np.vstack(all_real)
        
        return all_real_x, np.array(all_tissue)    
    
       

    def evaluating_test(self, data_real_train, data_real_test, genes_name_list, n_runs=10, pca = True, n_components= 50):

        corr_l = []
        recall_l = []
        precision_l = []
        
        pca_corr_l = []
        pca_recall_l = []
        pca_precision_l = []
        
        results_collection = {}
        
        for model_name in ['Logistic Regression', 'SVM', 'Random Forest', 'MLP']:
            results_collection[model_name] = {
                'reverse_validation_r_r_accuracy': [],
                'reverse_validation_f_r_accuracy': [],
                'reverse_validation_r_f_accuracy': [],
                'reverse_validation_r_r_precision': [],
                'reverse_validation_f_r_precision': [],
                'reverse_validation_r_f_precision': [],
                'data_augmentation_(f+r)_r_accuracy': [],
                'data_augmentation_(f+r)_r_precision': [],
                'detection_r_vs_f_accuracy': [],
                'detection_r_vs_f_precision': [],
                'detection_AUC': [],
                'pca_detection_r_vs_f_accuracy': [],
                'pca_detection_r_vs_f_precision': [],
                'pca_detection_AUC': [],
            
            }
            
        cl2_res_f_r = {}
        cl2_res_r_r ={}
        cl2_res_r_f ={}
  
        cl4_res_fr_r ={}
        
        cl5_r_vs_f = {}
        cl5_pca_r_vs_f = {}
        
        save = True
        
        alpha = 0.05
        kolmogorov_smirnov_results = []
        
        
 
        for run in range(n_runs):
            
            
            print('Run:', run)
            if n_runs > 1: 
                save = False
            
            data_real, data_gen, tissue_label= self.generate_samples(data_real_train)
            data_test, data_fake_test, tissue_label_test= self.generate_samples(data_real_test)
            
            print("tissue label train:", np.unique(tissue_label))
            print("tissue label test:", np.unique(tissue_label_test)) 
            
            real_df = pd.DataFrame(data_test)
            gen_df = pd.DataFrame(data_fake_test)
            print("genes_name_list", len(genes_name_list))
            print("real_df shape", real_df.shape)
            real_df.columns = genes_name_list #since they are in a range (0,.., after the generation)
            gen_df.columns = genes_name_list
            n = real_df.shape[1]  
            
            
            # PCs
            

            
            
            pca = PCA(n_components=n_components)
            pca_train_data = pca.fit_transform(data_real)
            pca_gen_data = pca.transform(data_gen)
            pca_data_real_test = pca.transform(data_test)
            pca_data_fake_test = pca.transform(data_fake_test)
            cumulative_explained_variance =  np.cumsum(pca.explained_variance_ratio_)[-1]
            print(f"explained train variance ratio : {cumulative_explained_variance} with {n_components} components ")

            
            
            #Unsupervised metrics 
            
            
            gamma_dx_dz = gamma_coef(data_test, data_fake_test)
            corr_l.append(gamma_dx_dz)
            print('corr:',gamma_dx_dz)
            prec, recall = get_precision_recall(torch.from_numpy(data_test), torch.from_numpy(data_fake_test))
            precision_l.append(prec)
            recall_l.append(recall)

            
            
            # Unsupervised metrics - PCA 
            
            if pca: 
                pca_gamma_dx_dz = gamma_coef(pca_data_real_test, pca_data_fake_test)
                pca_corr_l.append(gamma_dx_dz)
                print('corr:',gamma_dx_dz)
                pca_prec, pca_recall = get_precision_recall(torch.from_numpy(pca_data_real_test), torch.from_numpy(pca_data_fake_test))
                pca_precision_l.append(pca_prec)
                pca_recall_l.append(recall)
            
            
            
            # Reverse validation 
            
            print('Reverse validation ..')
            
            
            cl2_res_r_r = Classifiers(data_real, tissue_label, data_test, tissue_label_test, 'multi_res_r_r', self.model.results_dire, save = save)
    
            cl2_res_f_r = Classifiers(data_gen, tissue_label, data_test, tissue_label_test, 'multi_res_f_r', self.model.results_dire, save = save)
            cl2_res_r_f = Classifiers(data_real, tissue_label, data_fake_test, tissue_label_test, 'multi_res_r_f', self.model.results_dire, save = save)
            
            
            for model_name in cl2_res_f_r.keys():
                results_collection[model_name]['reverse_validation_r_r_accuracy'].extend(cl2_res_r_r[model_name]['accuracy'])
                results_collection[model_name]['reverse_validation_f_r_accuracy'].extend(cl2_res_f_r[model_name]['accuracy'])
                results_collection[model_name]['reverse_validation_r_f_accuracy'].extend(cl2_res_r_r[model_name]['accuracy'])
                
                
                
                results_collection[model_name]['reverse_validation_r_r_precision'].extend(cl2_res_r_r[model_name]['precision'])
                results_collection[model_name]['reverse_validation_f_r_precision'].extend(cl2_res_f_r[model_name]['precision'])
                results_collection[model_name]['reverse_validation_r_f_precision'].extend(cl2_res_r_r[model_name]['precision'])
               
           
            
            
            # Data Augmentation (training dataset = real train datset + train fake dataset, test dataset = real test dataset)
            
            augmented_train_data = np.vstack([data_real, data_gen]) 
            tissue_label_rf = np.tile(tissue_label, 2)
            
            print('Data Augmentation...')
            
            cl4_res_fr_r = Classifiers(augmented_train_data, tissue_label_rf, data_test, tissue_label_test, 'data_augmentation_(f+r)_r', self.model.results_dire, save = save)
            
            for model_name in cl4_res_fr_r.keys():
                results_collection[model_name]['data_augmentation_(f+r)_r_accuracy'].extend(cl4_res_fr_r[model_name]['accuracy'])
                results_collection[model_name]['data_augmentation_(f+r)_r_precision'].extend(cl4_res_fr_r[model_name]['precision'])
                
     
            # Detection (classification of real (0) vs fake(1))
  
            
            # np.save('/home/mongardi/Synth_data_small/data_real.npy', data_real)
            # np.save('/home/mongardi/Synth_data_small/data_gen.npy', data_gen)
            
            
            
            detection_train_data = shuffle(np.vstack([data_real, data_gen]), random_state= SEED)
            detection_train_labels =shuffle(np.array([0] * len(data_real) + [1] * len(data_gen)), random_state=SEED)
            detection_test_data = shuffle(np.vstack([data_test, data_fake_test]),random_state= SEED)
            detection_test_labels =shuffle(np.array([0] * len(data_test) + [1] * len(data_fake_test)), random_state=SEED)
            
            
            if pca: 
                pca_detection_train_data = shuffle(np.vstack([pca_train_data, pca_gen_data]), random_state= SEED)
                pca_detection_train_labels =shuffle(np.array([0] * len(data_real) + [1] * len(data_gen)), random_state=SEED)
                pca_detection_test_data = shuffle(np.vstack([pca_data_real_test, pca_data_fake_test]),random_state= SEED)
                pca_detection_test_labels =shuffle(np.array([0] * len(data_test) + [1] * len(data_fake_test)), random_state=SEED)
           

            
            
            
            cl5_r_vs_f = Classifiers(detection_train_data, detection_train_labels , detection_test_data, detection_test_labels, 'detection_r_vs_f', self.model.results_dire, save = save, detection = True)
            cl5_pca_r_vs_f = Classifiers(detection_train_data, detection_train_labels , detection_test_data, detection_test_labels, 'pca_detection_r_vs_f', self.model.results_dire, save = save, detection = True)
            
            
            for model_name in cl5_r_vs_f.keys():
                results_collection[model_name]['detection_r_vs_f_accuracy'].extend(cl5_r_vs_f[model_name]['accuracy'])
                results_collection[model_name]['detection_r_vs_f_precision'].extend(cl5_r_vs_f[model_name]['precision'])
                results_collection[model_name]['detection_AUC'].extend(cl5_r_vs_f[model_name]['auc'])
                
            for model_name in cl5_pca_r_vs_f.keys():
                results_collection[model_name]['pca_detection_r_vs_f_accuracy'].extend(cl5_pca_r_vs_f[model_name]['accuracy'])
                results_collection[model_name]['pca_detection_r_vs_f_precision'].extend(cl5_pca_r_vs_f[model_name]['precision'])
                results_collection[model_name]['pca_detection_AUC'].extend(cl5_pca_r_vs_f[model_name]['auc'])
                
            
            
            #TWO - SAMPLE KOLMOGOROV-SMIRNOV TEST 
            
            significant = []
            p_values_ks = {column: [] for column in real_df.columns}
            
            for column in real_df.columns:
                
    
                #print(".  .  . Testing for columns", column)
                ks_statistic, p_value = stats.ks_2samp(real_df.loc[:,column], gen_df.loc[:,column])
                
                p_values_ks[column] = p_value
                p_value_corrected = p_values_ks[column] * n
                
    
                if p_value_corrected < alpha: 
                    significant.append(True)
                else: 
                    significant.append(False)   
               
                
                    
            print(f"Number of columns that differ significantly between the real and generated data after correction at run {run}:", sum(significant))

            
            
            
    # results         
                     
        print('corr:', np.mean(corr_l), np.std(corr_l))
        print('precision:', np.mean(precision_l), np.std(precision_l))
        print('recall:', np.mean(recall_l), np.std(recall_l))
        
        
        print('pca_corr:', np.mean(pca_corr_l), np.std(pca_corr_l))
        print('pca_precision:', np.mean(pca_precision_l), np.std(pca_precision_l))
        print('pca_recall:', np.mean(pca_recall_l), np.std(pca_recall_l))
        
        for model_name, metrics in results_collection.items():
                reverse_val_r_r_acc_mean = np.mean(metrics['reverse_validation_r_r_accuracy'])
                reverse_val_r_r_acc_std = np.std(metrics['reverse_validation_r_r_accuracy'])
                reverse_val_r_r_prec_mean = np.mean(metrics['reverse_validation_r_r_precision'])
                reverse_val_r_r_prec_std = np.std(metrics['reverse_validation_r_r_precision'])
                
                
                reverse_val_f_r_acc_mean = np.mean(metrics['reverse_validation_f_r_accuracy'])
                reverse_val_f_r_acc_std = np.std(metrics['reverse_validation_f_r_accuracy'])
                reverse_val_f_r_prec_mean = np.mean(metrics['reverse_validation_f_r_precision'])
                reverse_val_f_r_prec_std = np.std(metrics['reverse_validation_f_r_precision'])
                
                
                reverse_val_r_f_acc_mean = np.mean(metrics['reverse_validation_r_f_accuracy'])
                reverse_val_r_f_acc_std  = np.std(metrics['reverse_validation_r_f_accuracy'])
                reverse_val_r_f_prec_mean = np.mean(metrics['reverse_validation_r_f_precision'])
                reverse_val_r_f_prec_std = np.std(metrics['reverse_validation_r_f_precision'])

                
                
                data_augmentation_fr_r_acc_mean = np.mean(metrics['data_augmentation_(f+r)_r_accuracy'])
                data_augmentation_fr_r_acc_std = np.std(metrics['data_augmentation_(f+r)_r_accuracy'])
                data_augmentation_fr_r_prec_mean = np.mean(metrics['data_augmentation_(f+r)_r_precision'])
                data_augmentation_fr_r_prec_std = np.std(metrics['data_augmentation_(f+r)_r_precision'])
                
                detection_r_f_acc_mean = np.mean(metrics['detection_r_vs_f_accuracy'])
                detection_r_f_acc_std = np.std(metrics['detection_r_vs_f_accuracy'])
                detection_r_f_prec_mean = np.mean(metrics['detection_r_vs_f_precision'])
                detection_r_f_prec_std = np.std(metrics['detection_r_vs_f_precision'])
                
                
                pca_detection_r_f_acc_mean = np.mean(metrics['pca_detection_r_vs_f_accuracy'])
                pca_detection_r_f_acc_std = np.std(metrics['pca_detection_r_vs_f_accuracy'])
                pca_detection_r_f_prec_mean = np.mean(metrics['pca_detection_r_vs_f_precision'])
                pca_detection_r_f_prec_std = np.std(metrics['pca_detection_r_vs_f_precision'])


                detection_r_f_auc_mean = np.mean(metrics['detection_AUC'])
                detection_r_f_auc_std = np.std(metrics['detection_AUC'])
                pca_detection_r_f_auc_mean = np.mean(metrics['pca_detection_AUC'])
                pca_detection_r_f_auc_std = np.std(metrics['pca_detection_AUC'])
             
                
                print(f'{model_name} - Reverse validation r_r accuracy : Mean = {reverse_val_r_r_acc_mean:.5f}, Std = {reverse_val_r_r_acc_std:.5f}')
                print(f'{model_name} - Reverse validation r_r precision : Mean = {reverse_val_r_r_prec_mean:.5f}, Std = {reverse_val_r_r_prec_std:.5f}')
                
                print(f'{model_name} - Reverse validation f_r accuracy : Mean = {reverse_val_f_r_acc_mean:.5f}, Std = {reverse_val_f_r_acc_std:.5f}')
                print(f'{model_name} - Reverse validation f_r precision : Mean = {reverse_val_f_r_prec_mean:.5f}, Std = {reverse_val_f_r_prec_std:.5f}')
                
                
                print(f'{model_name} - Reverse validation r_f accuracy : Mean = {reverse_val_r_f_acc_mean:.5f}, Std = {reverse_val_r_f_acc_std:.5f}')
                print(f'{model_name} - Reverse validation r_f precision : Mean = {reverse_val_r_f_prec_mean:.5f}, Std = {reverse_val_r_f_prec_std:.5f}')

                
                print(f'{model_name} - Data augmentation r+f_r accuracy : Mean = {data_augmentation_fr_r_acc_mean:.5f}, Std = {data_augmentation_fr_r_acc_std:.5f}')
                print(f'{model_name} - Data augmentation r+f_r precision : Mean = {data_augmentation_fr_r_acc_mean:.5f}, Std = {data_augmentation_fr_r_acc_std:.5f}')
                
                print(f'{model_name} - Detection_accuracy : Mean = {detection_r_f_acc_mean:.5f}, Std = {detection_r_f_acc_std:.5f}')
                print(f'{model_name} - Detection_precision : Mean = {detection_r_f_prec_mean:.5f}, Std = {detection_r_f_prec_std:.5f}')
                
                print(f'{model_name} - Pca_Detection_accuracy : Mean = {pca_detection_r_f_acc_mean:.5f}, Std = {pca_detection_r_f_acc_std:.5f}')
                print(f'{model_name} - Pca_Detection_precision : Mean = {pca_detection_r_f_prec_mean:.5f}, Std = {pca_detection_r_f_prec_std:.5f}')

                print(f'{model_name} - Detection_auc : Mean = {detection_r_f_auc_mean:.5f}, Std = {detection_r_f_auc_std:.5f}')
                print(f'{model_name} - Pca_Detection_auc : Mean = {pca_detection_r_f_auc_mean :.5f}, Std = {pca_detection_r_f_auc_std:.5f}')


    
        
    def real_real(self, data_real_train, data_real_test, genes_name_list, n_runs=10, pca = True, n_components= 50):

        results_collection={}
        
        for model_name in ['Logistic Regression', 'SVM', 'Random Forest', 'MLP']:
            results_collection[model_name] = {
                'reverse_validation_r_r_accuracy': [],
                'reverse_validation_r_r_precision': [],
            }
            

        cl2_res_r_r ={}

        

        
        save = True
       
        
 
        for run in range(n_runs):
            
            
            print('Run:', run)
            if n_runs > 1: 
                save = False
            
            data_real, data_gen, tissue_label= self.generate_samples(data_real_train)
            data_test, data_fake_test, tissue_label_test= self.generate_samples(data_real_test)
            
            print("tissue label train:", np.unique(tissue_label))
            print("tissue label test:", np.unique(tissue_label_test)) 
            

            
            
         
            
            
            cl2_res_r_r = Classifiers(data_real, tissue_label, data_test, tissue_label_test, 'multi_res_r_r', self.model.results_dire, save = save)
    
            
            
            
            for model_name in cl2_res_r_r.keys():
                results_collection[model_name]['reverse_validation_r_r_accuracy'].extend(cl2_res_r_r[model_name]['accuracy'])
               
                
                
                results_collection[model_name]['reverse_validation_r_r_precision'].extend(cl2_res_r_r[model_name]['precision'])
               
               
           
            
            
            
       
        
        for model_name, metrics in results_collection.items():
                reverse_val_r_r_acc_mean = np.mean(metrics['reverse_validation_r_r_accuracy'])
                reverse_val_r_r_acc_std = np.std(metrics['reverse_validation_r_r_accuracy'])
                reverse_val_r_r_prec_mean = np.mean(metrics['reverse_validation_r_r_precision'])
                reverse_val_r_r_prec_std = np.std(metrics['reverse_validation_r_r_precision'])
                
               
             
                
                print(f'{model_name} - Reverse validation r_r accuracy : Mean = {reverse_val_r_r_acc_mean:.5f}, Std = {reverse_val_r_r_acc_std:.5f}')
                print(f'{model_name} - Reverse validation r_r precision : Mean = {reverse_val_r_r_prec_mean:.5f}, Std = {reverse_val_r_r_prec_std:.5f}')
                
                
        
        
 # detection function
def detection(data_real, data_gen, data_real_test, data_fake_test):
    #data_real_test, data_fake_test, cancer_label_test, tissue_label_test= self.generate_samples(real_test)
            
    detection_train_data = shuffle(np.vstack([data_real, data_gen]), random_state= SEED)
    detection_train_labels =shuffle(np.array([0] * len(data_real) + [1] * len(data_gen)), random_state=SEED)      
    detection_test_data = shuffle(np.vstack([data_real_test, data_fake_test]),random_state= SEED)
    detection_test_labels =shuffle(np.array([0] * len(data_real_test) + [1] * len(data_fake_test)), random_state=SEED)
    cl5_r_vs_f = Classifiers(detection_train_data, detection_train_labels , detection_test_data, detection_test_labels, None, None, save = False, detection = True)

    results = {}
    for model_name in cl5_r_vs_f.keys():
        results[model_name] = {}
        results[model_name]['accuracy'] =  cl5_r_vs_f[model_name]['accuracy']
        results[model_name]['f1'] = cl5_r_vs_f[model_name]['f1_macro']
        results[model_name]['auc'] = cl5_r_vs_f[model_name]['auc']
    print(results)  
    return results 
        
    


# linear classifier
lr_model_args = {
            "random_state": SEED,
            "n_jobs": -1,
            "max_iter": 10000,
            "penalty": 'l2',
        }


svm_model_args = {
    "random_state": SEED,
    "max_iter": 10000,
}

rf_model_args = {
    "random_state": SEED,
    "n_jobs": -1,
    "n_estimators": 100,  
    "max_depth": None,  
}

mlp_model_args = {
    "random_state": SEED,
    "max_iter": 1000,
    "hidden_layer_sizes": (100,), 
    "activation": 'relu',  
    "solver": 'adam',
}


def Linear_classifier(X_train, y_train, X_test, y_test,  description , resulst_dire, name= None, save = True):
    print('Linear classifier')
    labels = np.unique(y_test)
    model = LogisticRegression(**model_args)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    df_result = show_single_class_evaluation(y_pred, y_test, labels, name, resulst_dire, save=save, description = description)
    save_features = True
    # if save_features:
        
    #     coeffs = model.coef_
    #     inds = np.argsort(abs(coeffs))
    #     np.save('/home/mongardi/Synth_data_small/inds.npy', inds)
    return df_result


def Classifiers(X_train, y_train, X_test, y_test,  description , resulst_dire, name= None, save = True, detection = False, cm = False):
        models = {
        'Logistic Regression': LogisticRegression(**lr_model_args),
        #'SVM': SVC(**svm_model_args),
        'Random Forest': RandomForestClassifier(**rf_model_args),
        'MLP': MLPClassifier(**mlp_model_args)
        }
        

        labels = np.unique(y_test)
        results = {}
        
        for model_name, model in models.items():
            print(f'Training {model_name}')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
          
            labels_p = np.unique(y_pred)
            
            #print([x for x in labels_p if x not in labels])
            
            
            
            description = f"{model_name}- {description}"
            df_result = show_single_class_evaluation(
            y_pred, y_test, labels, name, resulst_dire, save=save, description=description, detection = detection 
            )
        
            results[model_name] = df_result
            if cm == True: 
                cm = confusion_matrix(y_test, y_pred)
                
                plt.figure(figsize=(10, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

                plt.title(f'Training {model_name} - description {description}')
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
            
                filename = f'confusion_matrix_{model_name}_{description.replace(" ", "_")}.png'
                plt.savefig(filename)
            
                plt.show()
        return results       


def show_single_class_evaluation(y_pred: int, y_test: int, labels, name=None, results_dire=None, verbose=False, save =True, description='', detection = False):

    if verbose:
        print("Balanced accuracy: ", round(balanced_accuracy_score(y_test, y_pred), 5)) # not possible for single class
        print("Accuracy: ", round(accuracy_score(y_test, y_pred), 5)) # not possible for single class
        print('precision ', round(precision_score(y_test, y_pred, average="macro"), 5))
        print('recall ', round(recall_score(y_test, y_pred, average="macro"), 5))
        print('f1_macro ', round(f1_score(y_test, y_pred, average="macro"),5))
        print('f1_weighted ', round(f1_score(y_test, y_pred, average="weighted"),5))
        print("Precision: ", [round(i, 5) for i in precision_score(y_test, y_pred, average=None) ])
        print("Recall: ",  [round(i, 5) for i in recall_score(y_test, y_pred, average=None) ]) 
        print("F1 Score: ", [round(i, 5) for i in f1_score(y_test, y_pred, average=None) ]) 
        print('--------------------------------------------')
        


    
    dic_result = {}
    dic_result['balanced_accuracy'] = [round(balanced_accuracy_score(y_test, y_pred), 5)]
    dic_result['accuracy'] = [round(accuracy_score(y_test, y_pred), 5)]
    dic_result['precision'] = [round(precision_score(y_test, y_pred, average="macro"), 5)]
    dic_result['recall'] = [round(recall_score(y_test, y_pred, average="macro"), 5)]
    dic_result['f1_macro'] = [round(f1_score(y_test, y_pred, average="macro"),5)]
    dic_result['f1_weighted'] = [round(f1_score(y_test, y_pred, average="weighted"),5)]
    
    if detection:
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        dic_result['auc'] = [round(roc_auc,5)]
    
    for i in range(len(labels)):
        dic_result[str(labels[i])+'-precision'] =  round(precision_score(y_test, y_pred, average=None)[i], 5)
    for i in range(len(labels)):
        dic_result[str(labels[i])+'-recall'] =  round(recall_score(y_test, y_pred, average=None)[i], 5)
    for i in range(len(labels)):   
        dic_result[str(labels[i])+'-f1_score'] =  round(f1_score(y_test, y_pred, average=None)[i], 5)

    #df_result = pd.DataFrame.from_dict(dic_result)
    #df_result.to_csv(os.path.join(results_dire, name +'_output_detailed_scores.csv'), index=False)
    if save:
        #save_dictionary(dic_result, os.path.join(results_dire, name +'_output_detailed_scores',description))
        pass
    return dic_result

def save_dictionary(dictionary,filename,description):
    with open(filename + ".json", "w") as fp:
        fp.write("\n")
        fp.write(description)
        fp.write("\n")
        fp.write("\n")
        fp.write("\n")
        json.dump(dictionary, fp)
       
        print("Done writing dict into .json file")