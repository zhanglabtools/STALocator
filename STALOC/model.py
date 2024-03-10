# Modified from Portal:
# Zhao J, et al. (2022) Adversarial domain translation networks for integrating large-scale atlas-level single-cell datasets. Nature Computational Science 2(5):317-330.

import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import tables
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA, IncrementalPCA
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from tqdm import tqdm

from STALocator.networks import *
from STALocator.train import *

class Model(object):
    def __init__(self,
                 resolution = "low", # Resolution of ST dataset. Available options are: "low" and "high".
                 batch_size = 500, # Batch size.
                 train_epoch = 5000, # Epochs.
                 cut_steps = 0.5, # Switch location between integration network and localization network training.
                 seed = 1234, # Random seed.
                 npcs = 30, # Number of top PCs.
                 n_latent = 20, # Dimension of shared latent space on integration network.
                 n_coord = 2, # Dimension of spatial location.
                 sf_coord = 12000, # scaling factor of spatial location.
                 location = "spatial", # storing location of spatial location.
                 rad_cutoff = None, # The distance of the location cell from the nearest spot is used to filter out cells located outside the tissue section.
                 lambdaGAN = 1.0, # The weight of adversarial learning loss on integration network.
                 lambdacos = 20.0, # The weight of cosine similarity on integration network.
                 lambdaAE = 10.0, # The weight of auto-encoder consistency on integration network.
                 lambdaLA = 10.0, # The weight of latent alignment loss on integration network.
                 lambdaSWD = 5.0, # The weight of sliced Wasserstein distance on integration network.
                 lambdalat = 1.0, # The weight of spatial location fitting loss on localization network.
                 lambdarec = 0.01, # The weight of reconstruction loss on localization network.
                 model_path = "models", # Model save path.
                 data_path = "data", # Data save path.
                 result_path = "results", # Result save path.
                 ot = True, # Whether to perform minibatch optimal transport. Available options are: True and False.
                 verbose = True, # Whether to print running information. Available options are: True and False.
                 device = "cpu" # The device of model running. Specific graphic card should be specified if use GPU.
                 ):

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        if resolution not in ['low', 'high']:
            raise ValueError("Resolution must be either 'low' or 'high'.")

        if resolution == "high" and not ot:
            raise ValueError("If resolution is 'high', ot must be True.")

        self.resolution = resolution
        self.batch_size = batch_size
        self.train_epoch = train_epoch
        if self.resolution == "low":
            self.cut_steps = cut_steps
        elif self.resolution == "high":
            self.cut_steps = 1

        self.npcs = npcs
        self.n_latent = n_latent
        self.n_coord = n_coord
        self.sf_coord = sf_coord
        self.location = location
        self.rad_cutoff = rad_cutoff

        self.lambdaGAN = lambdaGAN
        self.lambdacos = lambdacos
        self.lambdaAE = lambdaAE
        self.lambdaLA = lambdaLA
        self.lambdaSWD = lambdaSWD

        self.lambdalat = lambdalat
        self.lambdarec = lambdarec

        self.margin = 5.0

        self.model_path = model_path
        self.data_path = data_path
        self.result_path = result_path

        self.ot = ot
        self.verbose = verbose
        self.device = device

    def preprocess(self,
                   adata_A_input, # Anndata object of scRNA-seq data
                   adata_B_input, # Anndata object of ST data
                   hvg_num = 4000, # Number of highly variable genes for each anndata
                   save_embedding = False # Save low-dimensional embeddings or not
                   ):

        adata_A_input.obs["batch"] = "scRNA-seq"
        adata_B_input.obs["batch"] = "ST"

        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()

        if self.verbose:
            print("Finding highly variable genes...")
        sc.pp.highly_variable_genes(adata_A, flavor='seurat_v3', n_top_genes=hvg_num)
        sc.pp.highly_variable_genes(adata_B, flavor='seurat_v3', n_top_genes=hvg_num)
        hvg_A = adata_A.var[adata_A.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_B = adata_B.var[adata_B.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        # hvg_total = hvg_A & hvg_B
        # hvg_total = list(set(hvg_A) & set(hvg_B))
        hvg_total = hvg_A.intersection(hvg_B)
        if len(hvg_total) < 100:
            raise ValueError("The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(hvg_total))

        if self.verbose:
            print("Normalizing and scaling...")
        sc.pp.normalize_total(adata_A, target_sum=1e4)
        sc.pp.log1p(adata_A)
        adata_A = adata_A[:, hvg_total]
        sc.pp.scale(adata_A, max_value=10)

        sc.pp.normalize_total(adata_B, target_sum=1e4)
        sc.pp.log1p(adata_B)
        adata_B = adata_B[:, hvg_total]
        sc.pp.scale(adata_B, max_value=10)

        adata_total = adata_A.concatenate(adata_B, index_unique=None)

        if self.verbose:
            print("Dimensionality reduction via PCA...")
        pca = PCA(n_components=self.npcs, svd_solver="arpack", random_state=0)
        adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)

        self.emb_A = adata_total.obsm["X_pca"][:adata_A.shape[0], :self.npcs].copy()
        self.emb_B = adata_total.obsm["X_pca"][adata_A.shape[0]:, :self.npcs].copy()
        self.coord_B = adata_B.obsm[self.location][:, :self.n_coord].copy()/self.sf_coord
        self.adata_total = adata_total
        self.adata_A_input = adata_A_input
        self.adata_B_input = adata_B_input

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if save_embedding:
            np.save(os.path.join(self.data_path, "lowdim_A.npy"), self.emb_A)
            np.save(os.path.join(self.data_path, "lowdim_B.npy"), self.emb_B)

    def train(self,
              num_projections = 500, # The same as sliced_wasserstein_distance function and _sliced_wasserstein_distance function.
              metric = 'correlation', # The same as trans_plan_b function.
              reg = 0.1, # The same as trans_plan_b function.
              numItermax = 10 # The same as trans_plan_b function.
              ):
        begin_time = time.time()
        if self.verbose:
            print("Begining time: ", time.asctime(time.localtime(begin_time)))
        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.D_A = discriminator(self.npcs).to(self.device)
        self.D_B = discriminator(self.npcs).to(self.device)
        self.E_s = encoder_site(self.n_latent, self.n_coord).to(self.device)
        self.D_s = decoder_site(self.n_latent, self.n_coord).to(self.device)
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        params_S = list(self.E_s.parameters()) + list(self.D_s.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.)
        optimizer_S = optim.Adam(params_S, lr=0.001, weight_decay=0.)
        params_D = list(self.D_A.parameters()) + list(self.D_B.parameters())
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.E_s.train()
        self.D_s.train()
        self.D_A.train()
        self.D_B.train()

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]

        if self.ot:
            plan = np.ones(shape=(N_A, N_B))
            plan = plan / (self.batch_size * self.batch_size)
            plan = torch.from_numpy(plan).float().to(self.device)

        for j in range(self.train_epoch):
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            c_B = torch.from_numpy(self.coord_B[index_B, :]).float().to(self.device)
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            m_B = self.E_s(z_B)
            s_Brecon = self.D_s(m_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)

            if j < int(self.train_epoch*self.cut_steps):
                optimizer_D.zero_grad()
                if j <= 5:
                    # Warm-up
                    loss_D_A = (torch.log(1 + torch.exp(-self.D_A(x_A))) + torch.log(
                        1 + torch.exp(self.D_A(x_BtoA)))).mean()
                    loss_D_B = (torch.log(1 + torch.exp(-self.D_B(x_B))) + torch.log(
                        1 + torch.exp(self.D_B(x_AtoB)))).mean()
                else:
                    loss_D_A = (torch.log(
                        1 + torch.exp(-torch.clamp(self.D_A(x_A), -self.margin, self.margin))) + torch.log(
                        1 + torch.exp(torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean()
                    loss_D_B = (torch.log(
                        1 + torch.exp(-torch.clamp(self.D_B(x_B), -self.margin, self.margin))) + torch.log(
                        1 + torch.exp(torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
                loss_D = loss_D_A + loss_D_B
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                loss_AE_A = torch.mean((x_Arecon - x_A) ** 2)
                loss_AE_B = torch.mean((x_Brecon - x_B) ** 2)
                loss_AE = loss_AE_A + loss_AE_B

                loss_cos_A = (1 - torch.sum(F.normalize(x_AtoB, p=2) * F.normalize(x_A, p=2), 1)).mean()
                loss_cos_B = (1 - torch.sum(F.normalize(x_BtoA, p=2) * F.normalize(x_B, p=2), 1)).mean()
                loss_cos = loss_cos_A + loss_cos_B

                loss_LA_AtoB = torch.mean((z_A - z_AtoB) ** 2)
                loss_LA_BtoA = torch.mean((z_B - z_BtoA) ** 2)
                loss_LA = loss_LA_AtoB + loss_LA_BtoA

                loss_SWD = sliced_wasserstein_distance(z_A, z_B, num_projections=num_projections, device=self.device)

                if self.ot:
                    plan_tmp = trans_plan_b(z_A, z_B, metric=metric, reg=reg, numItermax=numItermax, device=self.device)

                    coord_list = [[i, j] for i in index_A for j in index_B]
                    coord_list = np.array(coord_list)
                    plan[coord_list[:, 0], coord_list[:, 1]] = plan_tmp.reshape([self.batch_size * self.batch_size, ])

                if j <= 5:
                    # Warm-up
                    loss_GAN = (torch.log(1 + torch.exp(-self.D_A(x_BtoA))) + torch.log(
                        1 + torch.exp(-self.D_B(x_AtoB)))).mean()
                else:
                    loss_GAN = (torch.log(
                        1 + torch.exp(-torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin))) + torch.log(
                        1 + torch.exp(-torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()

                optimizer_G.zero_grad()
                loss_G = self.lambdaGAN * loss_GAN + self.lambdacos * loss_cos + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaSWD * loss_SWD
                loss_G.backward()
                optimizer_G.step()

                if not j % 500:
                    if self.verbose:
                        print("step %d, total_loss=%.4f, loss_D=%.4f, loss_GAN=%.4f, loss_AE=%.4f, loss_cos=%.4f, loss_LA=%.4f, loss_SWD=%.4f"
                              % (j, loss_G, loss_D, loss_GAN, loss_AE, loss_cos, loss_LA, loss_SWD))

            else:
                optimizer_S.zero_grad()

                loss_lat = loss1(m_B, c_B) + 0.1 * sliced_wasserstein_distance(m_B, c_B, num_projections=num_projections, device=self.device)
                loss_rec = loss2(s_Brecon, z_B) + 0.1 * loss1(s_Brecon, z_B)

                loss_S = self.lambdalat * loss_lat + self.lambdarec * loss_rec
                loss_S.backward()
                optimizer_S.step()

                if not j % 500:
                    if self.verbose:
                        print("step %d, loss_lat=%.4f, loss_rec=%.4f" % (j, loss_lat, loss_rec))


        end_time = time.time()
        if self.verbose:
            print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        if self.verbose:
            print("Training takes %.2f seconds" % self.train_time)

        if self.ot:
            self.plan = plan.detach().cpu().numpy()

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'D_A': self.D_A.state_dict(), 'D_B': self.D_B.state_dict(),
                 'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict(),
                 'E_s': self.E_s.state_dict(), 'D_s': self.D_s.state_dict()}

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))


    def eval(self,
             D_score = False, # Whether to output discriminated score.
             save_embedding = False, # Save low-dimensional embeddings or not.
             hvg_num = 4000, # Number of highly variable genes for each anndata.
             retain_prop = 1 # The proportion of cells mapped for each sample of ST dataset when data enhancement of extension.
             ):
        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.E_s = encoder_site(self.n_latent, self.n_coord).to(self.device)
        self.D_s = decoder_site(self.n_latent, self.n_coord).to(self.device)
        self.E_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_A'])
        self.E_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_B'])
        self.G_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_A'])
        self.G_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_B'])
        self.E_s.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_s'])
        self.D_s.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['D_s'])

        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        x_B = torch.from_numpy(self.emb_B).float().to(self.device)

        z_A = self.E_A(x_A)
        z_B = self.E_B(x_B)

        m_A = self.E_s(z_A)
        m_B = self.E_s(z_B)

        x_AtoB = self.G_B(z_A)
        x_BtoA = self.G_A(z_B)

        if self.resolution == "high":
            retain_cell = retain_prop * self.plan.T.shape[1]
            if retain_cell < 5:
                raise ValueError(
                    "The retained proportion is smaller than 5 (%d). Try to set a larger retain_prop." % retain_cell)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.adata_total.obsm['latent'] = self.latent
        self.data_Aspace = np.concatenate((self.emb_A, x_BtoA.detach().cpu().numpy()), axis=0)
        self.data_Bspace = np.concatenate((x_AtoB.detach().cpu().numpy(), self.emb_B), axis=0)

        if self.resolution == "low":
            self.map = np.concatenate((m_A.detach().cpu().numpy(), m_B.detach().cpu().numpy()), axis=0)*self.sf_coord
            self.map_A = m_A.detach().cpu().numpy()*self.sf_coord
            self.map_B = m_B.detach().cpu().numpy()*self.sf_coord

        self.adata_total.obs["batch"] = self.adata_total.obs["batch"].replace(["0", "1"], ["scRNA-seq", "ST"])

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Add map results, save adata_total
        if self.resolution == "low":
            self.adata_total.obsm['loc'] = self.map
            self.adata_B_input.obsm["loc"] = self.map_B
            self.adata_A_input.obsm["loc"] = self.map_A
        self.adata_total.write(os.path.join(self.result_path, "adata_total.h5ad"))

        # Normalization, save adata_sc and adata_ST
        sc.pp.highly_variable_genes(self.adata_A_input, flavor="seurat_v3", n_top_genes=hvg_num)
        sc.pp.normalize_total(self.adata_A_input, target_sum=1e4)
        sc.pp.log1p(self.adata_A_input)
        self.adata_A_input.write(os.path.join(self.result_path, "adata_sc.h5ad"))

        sc.pp.highly_variable_genes(self.adata_B_input, flavor="seurat_v3", n_top_genes=hvg_num)
        sc.pp.normalize_total(self.adata_B_input, target_sum=1e4)
        sc.pp.log1p(self.adata_B_input)
        self.adata_B_input.write(os.path.join(self.result_path, "adata_ST.h5ad"))

        # keep cells inside section, save adata_sc_keep
        if self.resolution == "low":
            dist_with_spot = cdist(self.adata_A_input.obsm["loc"], self.adata_B_input.obsm["spatial"])
            min_dist = np.min(dist_with_spot, axis=1)
            self.adata_A_keep = self.adata_A_input[min_dist <= self.rad_cutoff]
            if 'spatial' in self.adata_B_input.uns:
                self.adata_A_keep.uns["spatial"] = self.adata_B_input.uns["spatial"]
                self.adata_A_keep.obsm["spatial"] = self.adata_A_keep.obsm["loc"]
            self.adata_A_keep.write(os.path.join(self.result_path, "adata_sc_keep.h5ad"))
            if self.verbose:
                print("Localized scRNA-seq dataset has been saved!")

        # save OT plan
        if self.ot:
            self.plan_df = pd.DataFrame(self.plan, index=self.adata_A_input.obs.index,
                                        columns=self.adata_B_input.obs.index)
            self.plan_df.index = self.plan_df.index.values
            self.plan_df.to_csv(os.path.join(self.result_path, "trans_plan.csv"))

        # save latent with batch and celltype
        self.latent_df = pd.DataFrame(self.latent, index=self.adata_total.obs.index)
        self.latent_df.columns = ["latent_" + str(x) for x in range(1, self.n_latent + 1)]
        self.latent_df["batch"] = self.adata_total.obs["batch"]
        self.latent_df["cell_type"] = self.adata_total.obs["celltype"]
        self.latent_df.to_csv(os.path.join(self.result_path, "latent.csv"))

        # save cell type scores
        if self.ot:
            sc_celltype = self.latent_df[self.latent_df["batch"] == "scRNA-seq"]
            sc_celltype = sc_celltype["cell_type"].astype("str")
            cluster_name = sc_celltype.unique()
            self.cluster_score = np.zeros(shape=(len(cluster_name), self.adata_B_input.shape[0]))
            self.cluster_score = pd.DataFrame(self.cluster_score, index=cluster_name,
                                              columns=self.adata_B_input.obs.index)
            for i in cluster_name:
                self.cluster_score.loc[i, :] = np.mean(self.plan[np.where(sc_celltype == i), :][0], axis=0)
            self.cluster_score = self.cluster_score.T
            self.cluster_score.to_csv(os.path.join(self.result_path, "cluster_score.csv"))

        # save transfer label, maximum probability
        if self.resolution == "high":
            self.trans_label = self.cluster_score.apply(lambda x: get_max_index(x), axis=1)
            self.trans_label = self.trans_label.replace(range(len(self.trans_label.unique())), self.cluster_score.columns)
            self.trans_label = pd.DataFrame(self.trans_label)
            self.trans_label.columns = ["transfer_label"]
            self.trans_label.index = self.adata_B_input.obs.index
            self.trans_label.to_csv(os.path.join(self.result_path, "trans_label.csv"))

        # save transfer data, probability mapping
        if self.resolution == "high":
            retain_cell = retain_prop * self.plan.shape[1]
            plan_filt = self.plan * (np.argsort(np.argsort(self.plan)) >= self.plan.shape[1] - retain_cell)
            self.plan_norm = plan_filt.T / np.sum(plan_filt, 1)
            self.plan_norm = sp.csr_matrix(self.plan_norm)
            self.data_pm = self.plan_norm @ self.adata_A_input.X
            self.adata_ST_pm = sc.AnnData(X=self.data_pm, obs=self.adata_B_input.obs, var=self.adata_A_input.var, obsm=self.adata_B_input.obsm)
            self.adata_ST_pm.write(os.path.join(self.result_path, "adata_ST_pm.h5ad"))
            if self.verbose:
                print("Enhanced ST dataset has been saved!")

        if D_score:
            self.D_A = discriminator(self.npcs).to(self.device)
            self.D_B = discriminator(self.npcs).to(self.device)
            self.D_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['D_A'])
            self.D_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['D_B'])

            score_D_A_A = self.D_A(x_A)
            score_D_B_A = self.D_B(x_AtoB)
            score_D_B_B = self.D_B(x_B)
            score_D_A_B = self.D_A(x_BtoA)

            self.score_Aspace = np.concatenate((score_D_A_A.detach().cpu().numpy(), score_D_A_B.detach().cpu().numpy()), axis=0)
            self.score_Bspace = np.concatenate((score_D_B_A.detach().cpu().numpy(), score_D_B_B.detach().cpu().numpy()), axis=0)

        if save_embedding:
            np.save(os.path.join(self.result_path, "latent_A.npy"), z_A.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "latent_B.npy"), z_B.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "x_AtoB.npy"), x_AtoB.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "x_BtoA.npy"), x_BtoA.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "map_A.npy"), m_A.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "map_B.npy"), m_B.detach().cpu().numpy())


