# For the Stereo-Seq Data

import numpy as np
import scipy.sparse as sp
from .GEMVAE import GEMVAE
import tensorflow as tf
import pandas as pd
import scanpy as sc
import matplotlib as plt

def train_GEMVAE(adata1,adata2, 
                hidden_dims1=[512, 30],hidden_dims2=[512, 30],z_dim=30, alpha=0, n_epochs=500, lr=0.0001, key_added='MY_ARCH',
                gradient_clipping=5, nonlinear=True, weight_decay=0.0001,verbose=True, 
                random_seed=2020, pre_labels=None, pre_resolution1=0.2, pre_resolution2=0.2,
                save_attention=False, save_loss=False, save_reconstrction=False,
                kl_loss = 0.01,contrastive_loss = 10,recon_loss = 1,weight_decay_loss = 1,recon_loss_type = "ZINB",task=''
                ):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    alpha
        The weight of cell type-aware spatial neighbor network.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    nonlinear
        If True, the nonlinear avtivation is performed.
    weight_decay
        Weight decay for AdamOptimizer.
    pre_labels
        The key in adata.obs for the manually designate the pre-clustering results. Only used when alpha>0.
    pre_resolution
        The resolution parameter of sc.tl.louvain for the pre-clustering. Only used when alpha>0 and per_labels==None.
    save_attention
        If True, the weights of the attention layers are saved in adata.uns['GEMVAE_attention']
    save_loss
        If True, the training loss is saved in adata.uns['GEMVAE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['GEMVAE_ReX'].

    Returns
    -------
    AnnData
    """

    #tf.reset_default_graph()
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    if 'highly_variable' in adata1.var.columns:
        adata_Vars1 =  adata1[:, adata1.var['highly_variable']]
    else:
        adata_Vars1 = adata1
    X1 = pd.DataFrame(adata_Vars1.X[:, ].toarray(), index=adata_Vars1.obs.index, columns=adata_Vars1.var.index)

    
    if 'highly_variable' in adata2.var.columns:
        adata_Vars2 =  adata2[:, adata2.var['highly_variable']]
    else:
        adata_Vars2 = adata2
    if task =='SSC':
        X2 = pd.DataFrame(adata_Vars2.X[:, ].toarray(), index=adata_Vars2.obs.index, columns=adata_Vars2.var.index)
    else:
        X2 = pd.DataFrame(adata_Vars2.X[:, ], index=adata_Vars2.obs.index, columns=adata_Vars2.var.index)

    if verbose:
        print('Size of Input for gene data : ', adata_Vars1.shape)
        print('Size of Input for protein data : ',adata_Vars2.shape)


    cells1 = np.array(X1.index)
    cells_id_tran1 = dict(zip(cells1, range(cells1.shape[0])))
    if 'Spatial_Net' not in adata1.uns.keys():
        raise ValueError("Spatial_Net is not existed for gene Run Cal_Spatial_Net first!")

    Spatial_Net1 = adata1.uns['Spatial_Net']
    G_df1 = Spatial_Net1.copy()
    G_df1['Cell1'] = G_df1['Cell1'].map(cells_id_tran1)
    G_df1['Cell2'] = G_df1['Cell2'].map(cells_id_tran1)
    G1 = sp.coo_matrix((np.ones(G_df1.shape[0]), (G_df1['Cell1'], G_df1['Cell2'])), shape=(adata1.n_obs, adata1.n_obs))
    global G_tf1
    G_tf1 = prepare_graph_data(G1)


    cells2 = np.array(X2.index)
    cells_id_tran2 = dict(zip(cells2, range(cells2.shape[0])))
    if 'Spatial_Net' not in adata2.uns.keys():
        raise ValueError("Spatial_Net is not existed for protein Run Cal_Spatial_Net first!")
    Spatial_Net2 = adata1.uns['Spatial_Net']
    G_df2 = Spatial_Net2.copy()
    G_df2['Cell1'] = G_df2['Cell1'].map(cells_id_tran2)
    G_df2['Cell2'] = G_df2['Cell2'].map(cells_id_tran2)
    G2 = sp.coo_matrix((np.ones(G_df2.shape[0]), (G_df2['Cell1'], G_df2['Cell2'])), shape=(adata2.n_obs, adata2.n_obs))
    global G_tf2
    G_tf2 = prepare_graph_data(G2)


    tf.compat.v1.disable_eager_execution()
    trainer = GEMVAE(hidden_dims1=[X1.shape[1]] + hidden_dims1,hidden_dims2=[X2.shape[1]] + hidden_dims2, z_dim=z_dim,alpha=alpha, 
                    n_epochs=n_epochs, lr=lr, gradient_clipping=gradient_clipping, 
                    nonlinear=nonlinear,weight_decay=weight_decay, verbose=verbose, 
                    random_seed=random_seed, 
                    kl_loss=kl_loss,contrastive_loss=contrastive_loss,recon_loss=recon_loss,weight_decay_loss=weight_decay_loss,recon_loss_type=recon_loss_type                    
                    )
    

    print("START TRAIN")
    
    
    if alpha == 0:
        trainer(G_tf1, G_tf2, G_tf1,G_tf2, X1,X2)
        embeddings, attention1,attention2, loss, ReX1, ReX2= trainer.infer(G_tf1, G_tf2, G_tf1,G_tf2, X1,X2)
    else:
        G_df1 = Spatial_Net1.copy()
        G_df2 = Spatial_Net1.copy()
        if pre_labels==None:
            if verbose:
                print('------Pre-clustering Genes using louvain with resolution=%.2f' %pre_resolution1)
            sc.tl.pca(adata1, svd_solver='arpack')
            sc.pp.neighbors(adata1)
            sc.tl.louvain(adata1, resolution=pre_resolution1, key_added='expression_louvain_label1')
            pre_labels1 = 'expression_louvain_label1'

            if verbose:
                print('------Pre-clustering Protein using louvain with resolution=%.2f' %pre_resolution2)
            sc.tl.pca(adata2, svd_solver='arpack')
            sc.pp.neighbors(adata2)
            sc.tl.louvain(adata2, resolution=pre_resolution2, key_added='expression_louvain_label2')
            pre_labels2 = 'expression_louvain_label2'

        prune_G_df1 = prune_spatial_Net(G_df1, adata1.obs[pre_labels1])
        prune_G_df2 = prune_spatial_Net(G_df2, adata2.obs[pre_labels2])

        prune_G_df1['Cell1'] = prune_G_df1['Cell1'].map(cells_id_tran1)
        prune_G_df2['Cell1'] = prune_G_df2['Cell1'].map(cells_id_tran2)

        prune_G_df1['Cell2'] = prune_G_df1['Cell2'].map(cells_id_tran1)
        prune_G_df2['Cell2'] = prune_G_df2['Cell2'].map(cells_id_tran2)

        prune_G1 = sp.coo_matrix((np.ones(prune_G_df1.shape[0]), (prune_G_df1['Cell1'], prune_G_df1['Cell2'])))
        prune_G2 = sp.coo_matrix((np.ones(prune_G_df2.shape[0]), (prune_G_df2['Cell1'], prune_G_df2['Cell2'])))
        
        prune_G_tf1 = prepare_graph_data(prune_G1)
        prune_G_tf2 = prepare_graph_data(prune_G2)

        prune_G_tf1 = (prune_G_tf1[0], prune_G_tf1[1], G_tf1[2])
        prune_G_tf2 = (prune_G_tf2[0], prune_G_tf2[1], G_tf2[2])

        if task == 'SSC':
            # pre-clustering result genes
            plt.rcParams["figure.figsize"] = (5, 5)
            sc.pl.spatial(adata1, img_key="hires", color="expression_louvain_label1", size=15, title='gene pre-clustering result', spot_size=10)

            # pre-clustering result protein 
            plt.rcParams["figure.figsize"] = (5, 5)
            sc.pl.spatial(adata2, img_key="hires", color="expression_louvain_label2", size=15, title='protein pre-clustering result', spot_size=10)
        
        elif task == 'SPATIAL_SC':
            plt.rcParams["figure.figsize"] = (5, 5)
            plt.rcParams["figure.figsize"] = (5, 5)

        else:
            # pre-clustering result genes
            plt.rcParams["figure.figsize"] = (5, 5)
            sc.pl.spatial(adata1, img_key="hires", color="expression_louvain_label1", size=1.5, title='gene pre-clustering result', spot_size=1)

            # pre-clustering result protein 
            plt.rcParams["figure.figsize"] = (5, 5)
            sc.pl.spatial(adata2, img_key="hires", color="expression_louvain_label2", size=1.5, title='protein pre-clustering result', spot_size=1)


        trainer(G_tf1,G_tf2, prune_G_tf1,prune_G_tf2, X1,X2)
        embeddings, attention1 , attention2 , loss, ReX1, ReX2 = trainer.infer(G_tf1,G_tf2, prune_G_tf1,prune_G_tf2, X1,X2)

    global df
    cell_reps = pd.DataFrame(embeddings)
    df=cell_reps
    cell_reps.index = cells1

    adata1.obsm[key_added] = cell_reps.loc[adata1.obs_names, ].values
    if save_attention:
        adata1.uns['gene_attention'] = attention1
        adata1.uns['protein_attention'] = attention2

    if save_loss:
        adata1.uns['arch_loss'] = loss
        
    if save_reconstrction:
        ReX1 = pd.DataFrame(ReX1, index=X1.index, columns=X1.columns)
        ReX1[ReX1<0] = 0
        adata1.layers['arch_ReX1'] = ReX1.values
        ReX2 = pd.DataFrame(ReX2, index=X2.index, columns=X2.columns)
        ReX2[ReX2<0] = 0
        adata2.layers['arch_ReX2'] = ReX2.values

    return adata1




def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' %Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']==Graph_df['Cell2_label'],]
    print('%d edges after pruning.' %Graph_df.shape[0])
    return Graph_df


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)# self-loop
    #data =  adj.tocoo().data
    #adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)

def recovery_Imputed_Count(adata, size_factor):
    assert('ReX1' in adata.uns)
    temp_df = adata.uns['ReX1'].copy()
    sf = size_factor.loc[temp_df.index]
    temp_df = np.expm1(temp_df)
    temp_df = (temp_df.T * sf).T
    adata.uns['ReX_Count1'] = temp_df
    
    assert('ReX2' in adata.uns)
    temp_df = adata.uns['ReX2'].copy()
    sf = size_factor.loc[temp_df.index]
    temp_df = np.expm1(temp_df)
    temp_df = (temp_df.T * sf).T
    adata.uns['ReX_Count2'] = temp_df

    return adata