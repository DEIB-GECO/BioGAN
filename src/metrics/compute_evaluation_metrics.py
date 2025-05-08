from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from .distribution_distances import (compute_distribution_distances, 
                                                     compute_knn_real_fake, 
                                                     compute_logistic_real_fake,
                                                     compute_random_forest_real_fake,
                                                     compute_prdc)

n_components = 10

def compute_evaluation_metrics(data_real, 
                               data_gen, 
                               data_test,
                               data_fake_test,
                               #model_name,
                               nn=10, 
                               original_space=True, 
                               pca = True,
                               knn_pca=None, 
                               knn_data=None):  

  # Metric dictionary
    #print(f"Evaluating for {model_name}")
    print("Real", data_real.shape)
    print("Generated", data_gen.shape)
    
    metrics = {}

    # Compute Wasserstein distance and MMD metrics
    mmd_wasserstein = compute_distribution_distances(torch.tensor(data_real).float(), 
                                                     torch.tensor(data_gen).float())
    mmd_wasserstein_test = compute_distribution_distances(torch.tensor(data_test).float(), 
                                                     torch.tensor(data_fake_test).float())

    for metric in mmd_wasserstein:

        metrics[metric] = mmd_wasserstein[metric]
        metrics[metric + "_test"] = mmd_wasserstein_test[metric]

    # Compute KNN identity metrics
    auc_real_fake = compute_knn_real_fake(data_real, data_gen,
                                        data_test, data_fake_test, n_neighbors=nn)
                                         
    metrics["KNN results"] = auc_real_fake
    
    auc_real_fake = compute_logistic_real_fake(data_real, data_gen,
                                        data_test, data_fake_test, n_neighbors=nn)
                                         
    metrics["Logistic results"] = auc_real_fake
    
    auc_real_fake = compute_random_forest_real_fake(data_real, data_gen,
                                        data_test, data_fake_test, n_neighbors=nn)
    
    
    metrics["Random Forest"] = auc_real_fake
    
    # Compute PRDC metrics in original space
    density_and_coverage = compute_prdc(data_real, 
                                        data_gen, 
                                        nearest_k=nn)
    density_and_coverage_test = compute_prdc(data_test, 
                                        data_fake_test, 
                                        nearest_k=nn)
    for metric in density_and_coverage:
        metrics[metric] = density_and_coverage[metric]
        metrics[metric + "_test"] = density_and_coverage_test[metric]
        
    if pca:

        pca = PCA(n_components=n_components)
        pca_train_data = pca.fit_transform(data_real)
        pca_gen_data = pca.transform(data_gen)
        pca_data_real_test = pca.transform(data_test)
        pca_data_fake_test = pca.transform(data_fake_test)
        cumulative_explained_variance =  np.cumsum(pca.explained_variance_ratio_)[-1]
        print(f"explained train variance ratio : {cumulative_explained_variance} with {n_components} components ")

        # Compute Wasserstein distance and MMD metrics
        mmd_wasserstein_pca= compute_distribution_distances(torch.tensor(pca_train_data).float(), 
                                                    torch.tensor(pca_gen_data).float())
        mmd_wasserstein_pca_test = compute_distribution_distances(torch.tensor(pca_data_real_test).float(), 
                                                    torch.tensor(pca_data_fake_test).float())                                           
        for metric in mmd_wasserstein:
            metrics[metric + "_PCA"] = mmd_wasserstein_pca[metric]
            metrics[metric + "_PCA_test"] = mmd_wasserstein_pca_test[metric]
        auc_real_fake_pca = compute_knn_real_fake(pca_train_data, 
                                                    pca_gen_data,
                                                    pca_data_real_test, 
                                                    pca_data_fake_test, 
                                                    n_neighbors=nn)
   
        metrics["KNN PCA results"] = auc_real_fake_pca

        auc_real_fake_pca = compute_logistic_real_fake(pca_train_data, 
                                                    pca_gen_data,
                                                    pca_data_real_test, 
                                                    pca_data_fake_test, 
                                                    n_neighbors=nn)
   
        metrics["Logistic PCA results"] = auc_real_fake_pca
        
        
        auc_real_fake_pca = compute_random_forest_real_fake(pca_train_data, 
                                                    pca_gen_data,
                                                    pca_data_real_test, 
                                                    pca_data_fake_test, 
                                                    n_neighbors=nn)
   
        metrics["Random Forest results"] = auc_real_fake_pca

        # Compute PRDC metrics in PCA space
        density_and_coverage_pca = compute_prdc(pca_train_data, 
                                                pca_gen_data, 
                                                nearest_k=nn)
        density_and_coverage_pca_test = compute_prdc(pca_data_real_test, 
                                        pca_data_fake_test, 
                                        nearest_k=nn)

        for metric in density_and_coverage_pca:
            metrics[metric + "_PCA"] = density_and_coverage_pca[metric]
            metrics[metric + "_PCA_test"] = density_and_coverage_pca_test[metric]
    
    # # Train and evaluate KNN classifier for cell type classification on original data
    # if knn_data:
    #     y_pred = knn_data.predict(adata_gen.X.A)    
    #     accuracy = f1_score(np.array(adata_gen.obs[category_field]), y_pred, average="macro")
    #     cell_type_metrics["KNN category"] = accuracy
    
    # # Train and evaluate KNN classifier for cell type classification on PCA data
    # if knn_pca:
    #     y_pred = knn_pca.predict(adata_gen.obsm["X_pca"])
    #     accuracy = f1_score(adata_gen.obs[category_field], y_pred, average="macro")
    #     cell_type_metrics["KNN category PCA"] = accuracy
    
    return metrics