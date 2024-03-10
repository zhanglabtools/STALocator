# Modified from Portal:
# Zhao J, et al. (2022) Adversarial domain translation networks for integrating large-scale atlas-level single-cell datasets. Nature Computational Science 2(5):317-330.

import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import umap
from STALocator.model import *
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


def preprocess_dataset(adata_A_used,
                       adata_B_used,
                       hvg_num=4000,
                       npcs=30,
                       n_coord=2,
                       sf_coord=12000,
                       location = "spatial",
                       ):

    adata_A_input = adata_A_used.copy()
    adata_B_input = adata_B_used.copy()

    adata_A_input.obs["batch"] = "scRNA-seq"
    adata_B_input.obs["batch"] = "ST"

    adata_A = adata_A_input.copy()
    adata_B = adata_B_input.copy()

    sc.pp.highly_variable_genes(adata_A, flavor='seurat_v3', n_top_genes=hvg_num)
    sc.pp.highly_variable_genes(adata_B, flavor='seurat_v3', n_top_genes=hvg_num)
    hvg_A = adata_A.var[adata_A.var.highly_variable == True].sort_values(by="highly_variable_rank").index
    hvg_B = adata_B.var[adata_B.var.highly_variable == True].sort_values(by="highly_variable_rank").index
    # hvg_total = hvg_A & hvg_B
    # hvg_total = list(set(hvg_A) & set(hvg_B))
    hvg_total = hvg_A.intersection(hvg_B)

    sc.pp.normalize_total(adata_A, target_sum=1e4)
    sc.pp.log1p(adata_A)
    adata_A = adata_A[:, hvg_total]
    sc.pp.scale(adata_A, max_value=10)

    sc.pp.normalize_total(adata_B, target_sum=1e4)
    sc.pp.log1p(adata_B)
    adata_B = adata_B[:, hvg_total]
    sc.pp.scale(adata_B, max_value=10)

    adata_total = adata_A.concatenate(adata_B, index_unique=None)

    pca = PCA(n_components=npcs, svd_solver="arpack", random_state=0)
    adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)

    emb_A = adata_total.obsm["X_pca"][:adata_A.shape[0], :npcs].copy()
    emb_B = adata_total.obsm["X_pca"][adata_A.shape[0]:, :npcs].copy()
    coord_B = adata_B.obsm[location][:, :n_coord].copy() / sf_coord
    adata_total = adata_total
    adata_A_input = adata_A_input
    adata_B_input = adata_B_input

    return [emb_A, emb_B, coord_B, adata_total, adata_A_input, adata_B_input]


def find_parameters(adata_sc,
                    adata_ST,
                    resolution = "low",
                    cut_steps = 0.5,
                    seed_list = [1234],
                    hvg_num = 4000,
                    npcs = 30,
                    n_coord = 2,
                    location = "spatial",
                    sf_coord = None,
                    rad_cutoff = None,
                    lambdacos_list = [2,5,10],
                    lambdaSWD_list = [0,5],
                    lambdalat = 10,
                    lambdarec_list = [0,0.1],
                    batch_size = 50,
                    train_epoch = 10000,
                    metric = 'correlation',
                    reg = 0.1,
                    numItermax = 100,
                    k_cutoff = 5,
                    mixingmetric_subsample = True,
                    label = False,
                    train_adata = None,
                    train_spatial_obsm = "spatial",
                    train_label_obs = "celltype",
                    pred_spatial_obsm = "loc",
                    true_label_obs = "celltype",
                    data_path = "data_parameter",
                    model_path = "models_parameter",
                    result_path = "results_parameter",
                    verbose = False,
                    device='cpu'
                    ):

    if resolution == "low":

        df_results = pd.DataFrame(columns=['seed', 'lambdacos', 'lambdaSWD', 'lambdarec', 'mixing_metric', 'clustering_metric'])
        for seed in seed_list:
            for lambdacos in lambdacos_list:
                for lambdarec in lambdarec_list:
                    for lambdaSWD in lambdaSWD_list:
                        model = Model(
                            resolution="low",
                            batch_size=batch_size,
                            train_epoch=train_epoch,
                            cut_steps=cut_steps,
                            seed=seed,
                            sf_coord=sf_coord,
                            rad_cutoff=rad_cutoff,
                            lambdacos=lambdacos,
                            lambdaSWD=lambdaSWD,
                            lambdalat=lambdalat,
                            lambdarec=lambdarec,
                            data_path=data_path,
                            model_path=model_path,
                            result_path=result_path,
                            ot=False,
                            verbose=verbose,
                            device=device
                        )

                        process_data_list = preprocess_dataset(
                            adata_sc,
                            adata_ST,
                            hvg_num=hvg_num,
                            npcs=npcs,
                            n_coord=n_coord,
                            sf_coord=sf_coord,
                            location=location
                        )
                        model.emb_A = process_data_list[0]
                        model.emb_B = process_data_list[1]
                        model.coord_B = process_data_list[2]
                        model.adata_total = process_data_list[3]
                        model.adata_A_input = process_data_list[4]
                        model.adata_B_input = process_data_list[5]
                        model.train(metric=metric, reg=reg, numItermax=numItermax)
                        model.eval()

                        meta = pd.DataFrame(index=np.arange(model.emb_A.shape[0] + model.emb_B.shape[0]))
                        meta["method"] = ["A"] * model.emb_A.shape[0] + ["B"] * model.emb_B.shape[0]
                        mixing = calculate_mixing_metric(model.latent, meta, k=5, max_k=300, methods=list(set(meta.method)),
                                                        subsample=mixingmetric_subsample)

                        if label:
                            train_data = np.array(train_adata.obsm[train_spatial_obsm])
                            train_label = train_adata.obs[train_label_obs].astype("str")
                            knn = KNeighborsClassifier(n_neighbors=k_cutoff)
                            knn.fit(train_data, train_label)
                            pred_data = np.array(model.adata_A_keep.obsm[pred_spatial_obsm])
                            pred_label = knn.predict(pred_data)
                            true_label = model.adata_A_keep.obs[true_label_obs]
                            ari = adjusted_rand_score(true_label, pred_label)

                            new_row = {
                                'seed': seed,
                                'lambdacos': lambdacos,
                                'lambdaSWD': lambdaSWD,
                                'lambdarec': lambdarec,
                                'mixing_metric': mixing,
                                'clustering_metric': ari,
                            }

                        elif not label:
                            pred_data = np.array(model.adata_A_keep.obsm[pred_spatial_obsm])
                            true_label = model.adata_A_keep.obs[true_label_obs]
                            ssc = silhouette_score(pred_data, true_label)

                            new_row = {
                                'seed': seed,
                                'lambdacos': lambdacos,
                                'lambdaSWD': lambdaSWD,
                                'lambdarec': lambdarec,
                                'mixing_metric': mixing,
                                'clustering_metric': ssc,
                            }

                        df_results = df_results.append(new_row, ignore_index=True)
                        print("Test: seed=%d, lambdacos=%.1f, lambdaSWD=%.1f, lambdarec=%.1f"
                              % (seed, lambdacos, lambdaSWD, lambdarec))

    elif resolution == "high":

        df_results = pd.DataFrame(columns=['seed','lambdacos', 'lambdaSWD', 'mixing_metric'])
        for seed in seed_list:
            for lambdacos in lambdacos_list:
                for lambdaSWD in lambdaSWD_list:
                    model = Model(
                        resolution="high",
                        batch_size=batch_size,
                        train_epoch=train_epoch,
                        cut_steps=1,
                        seed=seed,
                        rad_cutoff=rad_cutoff,
                        lambdacos=lambdacos,
                        lambdaSWD=lambdaSWD,
                        data_path=data_path,
                        model_path=model_path,
                        result_path=result_path,
                        verbose=verbose,
                        device=device
                    )

                    process_data_list = preprocess_dataset(adata_sc, adata_ST, hvg_num=4000)
                    model.emb_A = process_data_list[0]
                    model.emb_B = process_data_list[1]
                    model.coord_B = process_data_list[2]
                    model.adata_total = process_data_list[3]
                    model.adata_A_input = process_data_list[4]
                    model.adata_B_input = process_data_list[5]
                    model.train(metric=metric, reg=reg, numItermax=numItermax)
                    model.eval()

                    meta = pd.DataFrame(index=np.arange(model.emb_A.shape[0] + model.emb_B.shape[0]))
                    meta["method"] = ["A"] * model.emb_A.shape[0] + ["B"] * model.emb_B.shape[0]
                    mixing = calculate_mixing_metric(model.latent, meta, k=5, max_k=300, methods=list(set(meta.method)),
                                                     subsample=mixingmetric_subsample)

                    new_row = {
                        'seed': seed,
                        'lambdacos': lambdacos,
                        'lambdaSWD': lambdaSWD,
                        'mixing_metric': mixing,
                    }

                    df_results = df_results.append(new_row, ignore_index=True)
                    print("Test: seed=%d, lambdacos=%.1f, lambdaSWD=%.1f" % (seed, lambdacos, lambdaSWD))

    return df_results


def calculate_mixing_metric(data, meta, methods, k=5, max_k=300, subsample=True):
    if subsample:
        if data.shape[0] >= 1e4:
            np.random.seed(1234)
            subsample_idx = np.random.choice(data.shape[0], 10000, replace=False)
            data = data[subsample_idx]
            meta = meta.iloc[subsample_idx]
            meta.index = np.arange(len(subsample_idx))
    lowdim = data

    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='kd_tree').fit(lowdim)
    _, indices = nbrs.kneighbors(lowdim)
    indices = indices[:, 1:]
    mixing = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        if len(np.where(meta.method[indices[i, :]] == methods[0])[0]) > k-1:
            mixing[i, 0] = np.where(meta.method[indices[i, :]] == methods[0])[0][k-1]
        else: mixing[i, 0] = max_k - 1
        if len(np.where(meta.method[indices[i, :]] == methods[1])[0]) > k-1:
            mixing[i, 1] = np.where(meta.method[indices[i, :]] == methods[1])[0][k-1]
        else: mixing[i, 1] = max_k - 1
    return np.mean(np.median(mixing, axis=1) + 1)
