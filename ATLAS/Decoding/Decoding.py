"""Classification of biospatial units in TMG analysis

This module contains the main classes related to the task of classification of biospatial units (i.e. cells, isozones, regions) into type. 
The module is composed of a class hierarchy of classifiers. The "root" of this heirarchy is an abstract base class "Classifier" that has two 
abstract methods (train and classify) that any subclass will have to implement. 

"""

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from pynndescent import NNDescent
from tqdm import tqdm
import pandas as pd
from collections import Counter
import logging
import abc
from signal import valid_signals
import string
import numpy as np
import itertools
from datetime import datetime
from multiprocessing import Pool
from scipy.stats import norm
from scipy.stats import entropy, mode, median_abs_deviation
from scipy.spatial.distance import squareform
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score 
from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pynndescent 
import anndata
import os
import torch
import math
import matplotlib.colors as mcolors
from ATLAS.Utils import fileu
from ATLAS.Utils import tmgu
from ATLAS.Utils import pathu
from ATLAS.Utils import basicu
from functools import partial
from scipy.ndimage import gaussian_filter
import pickle
from tqdm import trange
import gc
from sklearn.feature_selection import f_classif


class KNN(object):
    def __init__(self,train_k=50,predict_k=500,max_distance=np.inf,metric='euclidean',verbose=False,weighted=False):
        self.train_k = train_k
        self.predict_k = predict_k
        self.max_distance = max_distance
        self.metric = metric
        self.verbose = verbose
        self.weighted = weighted
        self.homogeneous = False

    def fit(self,X,y):
        if np.unique(y).shape[0]==1:
            self.cts = np.unique(y)
            self.homogeneous = True
        else:
            self.homogeneous = False
            if self.weighted:
                F, p = f_classif(X, y)
                self.weights = F
                X = X*self.weights
            self.feature_tree_dict = {}
            self.feature_tree_dict['labels'] = y
            self.feature_tree_dict['tree'] = NNDescent(X, metric=self.metric, n_neighbors=self.train_k,n_trees=10,verbose=self.verbose)
            self.cts = np.array(sorted(np.unique(self.feature_tree_dict['labels'])))
            self.converter = dict(zip(self.cts,np.array(range(self.cts.shape[0]))))
            self.feature_tree_dict['labels_index'] = np.array([self.converter[i] for i in y])

    def predict(self,X,y=None):
        if self.homogeneous:
            return self.cts[0]*np.ones(X.shape[0])
        else:
            if self.weighted:
                X = X*self.weights
            if not isinstance(y,type(None)):
                self.feature_tree_dict['labels'] = y
                self.converter = dict(zip(self.cts,np.array(range(self.cts.shape[0]))))
                self.feature_tree_dict['labels_index'] = np.array([self.converter[i] for i in y])
            self.cts = np.array(sorted(np.unique(self.feature_tree_dict['labels'])))

            neighbors,distances = self.feature_tree_dict['tree'].query(X,k=self.predict_k)
            neighbor_types = self.feature_tree_dict['labels_index'][neighbors]
            neighbor_types[distances>self.max_distance]==-1
            likelihoods = torch.zeros([X.shape[0],self.cts.shape[0]])
            for cell_type in self.cts:
                likelihoods[:,self.converter[cell_type]] = torch.sum(1*torch.tensor(neighbor_types==self.converter[cell_type]),axis=1)
            return self.cts[likelihoods.max(1).indices]
    
    def predict_proba(self,X,y=None):
        if self.homogeneous:
            return np.ones([X.shape[0],1])
        else:
            if self.weighted:
                X = X*self.weights
            if not isinstance(y,type(None)):
                self.feature_tree_dict['labels'] = y
                self.converter = dict(zip(self.cts,np.array(range(self.cts.shape[0]))))
                self.feature_tree_dict['labels_index'] = np.array([self.converter(i) for i in y])
            self.cts = np.array(sorted(np.unique(self.feature_tree_dict['labels'])))

            neighbors,distances = self.feature_tree_dict['tree'].query(X,k=self.predict_k)
            neighbor_types = self.feature_tree_dict['labels_index'][neighbors]
            neighbor_types[distances>self.max_distance]==-1
            likelihoods = torch.zeros([X.shape[0],self.cts.shape[0]])
            for cell_type in self.cts:
                likelihoods[:,self.converter[cell_type]] = torch.sum(1*torch.tensor(neighbor_types==self.converter[cell_type]),axis=1)
            total = torch.sum(likelihoods,axis=1,keepdims=True)
            total[total==0] = 1
            likelihoods = likelihoods/total
            return likelihoods.numpy()
    
    @property
    def classes_(self):
        return self.cts


class KDESpatialPriors(Object):
    def __init__(self,
    ref='/scratchdata2/MouseBrainAtlases/MouseBrainAtlases_V0/Allen/',
    ref_levels=['class', 'subclass'],neuron=None,kernel = (0.25,0.1,0.1),
    border=1,binsize=0.1,bins=None,gates=None,types=None,symetric=False):
        self.out_path = f"/scratchdata1/KDE_kernel_{kernel[0]}_{kernel[1]}_{kernel[2]}_border_{border}_binsize_{binsize}_level_{ref_levels[-1]}_neuron_{neuron}_symetric_{symetric}.pkl"
        self.symetric = symetric
        if os.path.exists(self.out_path):
            temp = pickle.load(open(self.out_path,'rb'))
            self.typedata = temp.typedata
            self.types = temp.types
            self.ref_levels = temp.ref_levels
            self.neuron = temp.neuron
            self.kernel = temp.kernel
            self.bins = temp.bins
            self.gates = temp.gates
            self.border = temp.border
            self.binsize = temp.binsize
            self.ref = temp.ref
            self.converters = temp.converters
        else:
            if isinstance(ref,str):
                self.ref = TissueGraph.TissueMultiGraph(basepath = ref, input_df = None, redo = False).Layers[0].adata
            else:
                self.ref = ref

            self.ref_levels = ref_levels
            self.neuron = neuron
            self.kernel = kernel
            self.bins = bins
            self.gates = gates
            self.border = border
            self.binsize = binsize
            self.types = types
            self.typedata = None

    def train(self,dim_labels=['x_ccf','y_ccf','z_ccf']):
        """ check if types in self """
        if isinstance(self.typedata,type(None)):
            binsize = self.binsize
            border = self.border

            if self.symetric:
                dim = [i for i in dim_labels if 'z' in i][0]
                center = 5.71 #ccf_z'
                adata = self.ref.copy()
                # labels = dim_labels
                # labels.extend(self.ref_levels)
                # adata.obs = adata.obs[labels].copy()
                flipped_adata = adata.copy()
                flipped_adata.obs[dim] = center + (-1*(adata.obs[dim] - center))
                adata = anndata.concat([adata,flipped_adata])
                self.ref = adata


            XYZ = np.array(self.ref.obs[dim_labels])

            if isinstance(self.gates,type(None)):
                self.gates = []
                self.bins = []
                for dim in range(3):
                    vmin  = binsize*int((np.min(XYZ[:,dim])-border)/binsize)
                    vmax = binsize*int((np.max(XYZ[:,dim])+border)/binsize)
                    g = np.linspace(vmin,vmax,int((vmax-vmin)/binsize)+1)
                    self.gates.append(g)
                    self.bins.append(g[:-1]+binsize/2)
            bins = self.bins
            gates = self.gates

            labels = np.array(self.ref.obs[self.ref_levels[-1]])
            types = np.unique(labels)
            if isinstance(self.types,type(None)):
                if isinstance(self.neuron,bool):
                    if self.neuron:
                        print('Using Only Neurons')
                        types = np.array([i for i in types if not 'NN' in i])
                    else:
                        print('Using Only Non Neurons')
                        types = np.array([i for i in types if 'NN' in i])
                    # print(f" Using these Types only {types}")
            else:
                types = self.types
            self.types = types
            typedata = np.zeros([bins[0].shape[0],bins[1].shape[0],bins[2].shape[0],types.shape[0]],dtype=np.float16)
            for i in trange(types.shape[0],desc='Calculating Spatial KDE'):
                label = types[i]
                m = labels==label
                if np.sum(m)==0:
                    continue
                hist, edges = np.histogramdd(XYZ[m,:], bins=gates)
                # stk = gaussian_filter(hist,(0.5/binsize,0.25/binsize,0.25/binsize))
                # stk = gaussian_filter(hist,(0.25/binsize,0.1/binsize,0.1/binsize))
                stk = gaussian_filter(hist,(i/binsize for i in self.kernel))
                typedata[:,:,:,i] = stk
            density = np.sum(typedata,axis=-1,keepdims=True)
            density[density==0] = 1
            typedata = typedata/density
            self.typedata = typedata

            self.converters = {}
            for level in self.ref_levels:
                if level==self.ref_levels[-1]:
                    continue
                self.converters[level] = dict(zip(self.ref.obs[self.ref_levels[-1]],self.ref.obs[level]))
            pickle.dump(self,open(self.out_path,'wb'))

    def convert_priors(self,priors,level):
        converter = self.converters[level]
        types = np.unique([item for key,item in converter.items()])
        updated_priors = np.zeros([priors.shape[0],types.shape[0]])
        for i,label in enumerate(types):
            m = np.array([converter[key]==label for key in self.types])
            updated_priors[:,i] = np.sum(priors[:,m],axis=1)
        return updated_priors,types
        
    def classify(self,measured,level='subclass',dim_labels=['ccf_x','ccf_y','ccf_z']):
        XYZ = np.array(measured.obs[dim_labels])
        XYZ_coordinates = XYZ.copy()
        for dim in range(3):
            XYZ_coordinates[:,dim] = (XYZ_coordinates[:,dim]-self.bins[dim][0])/(self.bins[dim][1]-self.bins[dim][0])
        XYZ_coordinates = XYZ_coordinates.astype(int)

        priors = self.typedata[XYZ_coordinates[:,0],XYZ_coordinates[:,1],XYZ_coordinates[:,2],:]
        types = self.types

        if level!=self.ref_levels[-1]:
            priors,types = self.convert_priors(priors,level)
        return priors,types

import pynndescent
import leidenalg
import igraph as ig

class graphLeiden(Classifier):
    def __init__(self,adata,verbose=True):
        self.adata = adata
        self.verbose = verbose
        self.train()
        
    def train(self):
        adata = self.adata
        X = np.array(adata.layers['classification_space'].copy()).copy()
        fileu.update_user(f"Building Feature Graph",verbose=self.verbose)
        G,knn = tmgu.build_knn_graph(X,metric='correlation')
        self.G = G

    def classify(self,resolution=5):
        adata = self.adata
        fileu.update_user(f"Unsupervised Clustering",verbose=self.verbose)
        TypeVec = self.G.community_leiden(resolution=resolution,objective_function='modularity').membership
        # Convert to PyTorch tensor
        adata.obs['leiden'] =np.array(TypeVec).astype(str)
        cts = np.array(adata.obs['leiden'].unique())
        colors = np.random.choice(np.array(list(mcolors.XKCD_COLORS.keys())),cts.shape[0],replace=False)
        pallette = dict(zip(cts, colors))
        adata.obs['leiden_colors'] = adata.obs['leiden'].map(pallette)
        if self.verbose:
                sections = np.unique(adata.obs['Slice'])
                x = adata.obs['ccf_z']
                y = adata.obs['ccf_y']
                c = np.array(adata.obs[f"leiden_colors"])
                cl = np.array(adata.obs[f"leiden"])
                pallette = dict(zip(cl,c))
                cts = np.unique(cl)
                n_columns = np.min([5,sections.shape[0]])
                n_rows = math.ceil((1+sections.shape[0])/n_columns)
                fig,axs  = plt.subplots(n_rows,n_columns,figsize=[5*n_columns,5*n_rows])
                axs = axs.ravel()
                for ax in axs:
                    ax.axis('off')
                plt.suptitle(f"Unsupervised Classification res:{resolution}")
                for idx,section in enumerate(sections):
                    m = np.isin(np.array(adata.obs['Slice']),[section])
                    ax = axs[idx]
                    ax.scatter(x[m],y[m],s=0.01,c=c[m],marker='x')
                    ax.set_title(section)
                    ax.set_aspect('equal')
                    ax.axis('off')
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pallette[key], markersize=10, label=key) for key in cts]
                axs[idx+1].legend(handles=handles, loc='center',ncol=3, fontsize=8)
                axs[idx+1].axis('off')
                plt.show()
        return adata

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.cm as cm
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression

def merge_labels_correlation(X, y, correlation_threshold=0.9):
    # Step 1: Aggregate X by labels in y
    X = np.array(X)
    y = np.array(y).ravel()
    unique_labels = np.unique(y)
    averaged_data = {label: np.nanmedian(X[y == label, :], axis=0) for label in unique_labels}

    # Step 2: Create a DataFrame from the aggregated data
    df = pd.DataFrame(averaged_data).T  # Transpose to have labels as rows

    # print(df)

    # Step 3: Calculate pairwise correlations
    correlations = df.T.corr()
    # sns.clustermap(correlations, cmap='jet')
    condensed_distance_matrix = squareform(np.abs(correlations - 1), checks=False)

    # Step 4: Perform hierarchical clustering
    Z = linkage(condensed_distance_matrix, method='ward')
    clusters = fcluster(Z, t=1 - correlation_threshold, criterion='distance')

    # Step 5: Create a mapping between column names and their respective clusters
    column_clusters = {col: cluster for col, cluster in zip(df.index, clusters)}

    # Step 6: Generate colors for the clusters
    cts = np.array(np.unique(clusters))
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i / len(cts)) for i in range(len(cts))]
    pallette = dict(zip(cts, colors))

    # Step 7: Map the original labels to the new clusters
    new_labels = np.array(pd.Series(y).map(column_clusters)).astype(int)
    new_colors = np.array(pd.Series(new_labels).map(pallette))

    X = np.array(X)
    y = np.array(new_labels).ravel()
    unique_labels = np.unique(y)
    averaged_data = {label: np.nanmedian(X[y == label, :], axis=0) for label in unique_labels}


    # Step 2: Create a DataFrame from the aggregated data
    df = pd.DataFrame(averaged_data).T  # Transpose to have labels as rows
    # print(df)

    # Step 3: Calculate pairwise correlations
    correlations = df.T.corr()
    # sns.clustermap(correlations, cmap='jet')

    return new_labels, new_colors

class SingleCellAlignmentLeveragingExpectations(): 
    def __init__(self,measured,complete_reference='allen_wmb_tree',ref_level='subclass',verbose=True,visualize=False):
        self.verbose = verbose
        self.complete_reference = complete_reference
        self.measured = measured.copy()
        self.ref_level = ref_level
        self.visualize = visualize
        self.model = LogisticRegression(max_iter=1000) 
        self.likelihood_only = False
        self.prior_only = False

    def update_user(self,message):
        fileu.update_user(message,verbose=self.verbose,logger='SCALE')

    def unsupervised_clustering(self):
        # has a tendency to oversplit large cell types
        self.update_user("Running unsupervised clustering")
        X = self.measured.layers['classification_space'].copy()
        G,knn = tmgu.build_knn_graph(X,metric='correlation')
        TypeVec = G.community_leiden(resolution=10,objective_function='modularity').membership
        unsupervised_labels = np.array(TypeVec).astype(str)
        print(np.unique(unsupervised_labels).shape)
        unsupervised_labels,colors = merge_labels_correlation(X,unsupervised_labels)
        cts = np.array(np.unique(unsupervised_labels))
        # Generate colors evenly spaced across the jet colormap
        cmap = plt.get_cmap('gist_ncar')
        colors = [cmap(i / cts.shape[0]) for i in range(cts.shape[0])]
        np.random.shuffle(colors)
        # colors = np.random.choice(np.array(list(mcolors.XKCD_COLORS.keys())),cts.shape[0],replace=False)
        pallette = dict(zip(cts, colors))
        unsupervised_colors = np.array(pd.Series(unsupervised_labels).map(pallette))
        
        self.measured.obs['leiden'] = unsupervised_labels
        self.measured.obs['leiden_color'] = unsupervised_colors
        self.visualize_measured('leiden_color','Unsupervised Clustering')

    def calculate_spatial_priors(self):
        self.update_user("Loading Spatial Priors Class")
        kdesp = KDESpatialPriors(ref_levels=[self.ref_level],neuron=None,kernel=(0.25,0.1,0.1))
        kdesp.train()
        self.update_user("Calculating Spatial priors")
        priors = {}
        priors,types = kdesp.classify(self.measured, level=self.ref_level,dim_labels=['ccf_x','ccf_y','ccf_z'])
        priors[np.sum(priors,axis=1)==0,:] = 1 # if all zeros make it uniform
        priors = {'columns':types,'indexes':np.array(self.measured.obs.index),'matrix':priors.astype(np.float32)}

        self.priors = priors

    def load_reference(self):
        if isinstance(self.complete_reference,str):
            self.update_user("Loading Reference Data")
            self.complete_reference = anndata.read(pathu.get_path(self.complete_reference, check=True))
        
        shared_var = list(self.complete_reference.var.index.intersection(self.measured.var.index))
        self.reference = self.complete_reference[:,np.isin(self.complete_reference.var.index,shared_var)].copy()
        self.measured = self.measured[:,np.isin(self.measured.var.index,shared_var)]

        # Filter and reindex the reference and measured data to ensure the order matches
        self.reference = self.reference[:, shared_var].copy()
        self.measured = self.measured[:, shared_var].copy()

        self.update_user("Resampling Reference Data")
        # Balance Reference to Section Area
        Nbases = self.measured.shape[1]
        idxes = []
        # total_cells = np.min([self.measured.shape[0],500000])
        total_cells = 500000
        weights = np.mean(self.priors['matrix'],axis=0)
        weights = weights/weights.sum()
        for i,label in enumerate(self.priors['columns']):
            n_cells = int(total_cells*weights[i])
            if n_cells>10:
                m = self.reference.obs[self.ref_level]==label
                temp = np.array(self.reference.obs[m].index)
                if temp.shape[0]>0:
                    if np.sum(m)>n_cells:
                        idxes.extend(list(np.random.choice(temp,n_cells,replace=False)),)
                    else:
                        idxes.extend(list(np.random.choice(temp,n_cells)))
            # else:
            #     self.update_user(f"Removing {label} from Reference too few cells {n_cells}")
                
        self.reference = self.reference[idxes,:].copy()

        # add neuron annotation to reference
        self.unique_labels = np.array(self.reference.obs[self.ref_level].unique())
        self.neuron_labels = np.array([i for i in self.unique_labels if not 'NN' in i])
        self.non_neuron_labels = np.array([i for i in self.unique_labels if 'NN' in i])
        converter = {True:'Neuron',False:'Non_Neuron'}
        self.neuron_converter = {ct:converter[ct in self.neuron_labels] for ct in self.unique_labels}
        self.neuron_color_converter = {'Neuron':'k','Non_Neuron':'r'}
        self.reference.obs['neuron'] = self.reference.obs[self.ref_level].map(self.neuron_converter)
        self.reference.obs['neuron_color'] = self.reference.obs['neuron'].map(self.neuron_color_converter)

        self.ref_level_color_converter = dict(zip(self.reference.obs[self.ref_level],self.reference.obs[f"{self.ref_level}_color"]))

        # Perform Normalization
        self.reference.layers['raw'] = self.reference.X.copy()
        self.update_user("Normalizing Reference Data cell wise")
        self.reference.layers['normalized'] = basicu.normalize_fishdata_robust_regression(self.reference.X.copy())
        self.reference.layers['classification_space'] = self.reference.layers['normalized'].copy()
        self.update_user("Normalizing Reference Data bit wise")
        medians = np.zeros(self.reference.layers['classification_space'].shape[1])
        stds = np.zeros(self.reference.layers['classification_space'].shape[1])
        for i in range(self.reference.layers['classification_space'].shape[1]):
            vmin,vmax = np.percentile(self.reference.layers['classification_space'][:,i],[5,95])
            ref_mask = (self.reference.layers['classification_space'][:,i]>vmin)&(self.reference.layers['classification_space'][:,i]<vmax)
            medians[i] = np.median(self.reference.layers['classification_space'][ref_mask,i])
            stds[i] = np.std(self.reference.layers['classification_space'][ref_mask,i])
            self.reference.layers['classification_space'][:,i] = (self.reference.layers['classification_space'][:,i]-medians[i])/stds[i]
        self.ref_medians = medians
        self.ref_stds = stds

        self.update_user("Building Reference tree")
        self.feature_tree_dict = {}
        self.feature_tree_dict['labels'] = np.array(self.reference.obs.index.copy())
        self.feature_tree_dict['tree'] = NNDescent(self.reference.layers['classification_space'], metric='euclidean', n_neighbors=15,n_trees=10,verbose=self.verbose)

    def unsupervised_neuron_annotation(self):
        # Initial harmonization
        self.update_user("Performing Initial harmonization")
        self.measured.layers['classification_space'] = basicu.zscore_matching(self.reference.layers['classification_space'].copy(),self.measured.layers['normalized'].copy())
        self.visualize_layers('classification_space',measured_color='leiden_color',reference_color=f"{self.ref_level}_color",reference_layer = 'classification_space')

        self.update_user('Generateing Unsupervised Vectors')
        cts = self.measured.obs['leiden'].unique()
        X = np.zeros([len(cts),self.measured.shape[1]])
        for i,ct in enumerate(cts):
            m = self.measured.obs['leiden']==ct
            X[i,:] = np.median(self.measured.layers['classification_space'][m,:],axis=0)
        self.update_user("Querying Reference tree")
        neighbors,distances = self.feature_tree_dict['tree'].query(X,k=25)
        numerical_converter = {'Neuron':0,'Non_Neuron':1}
        referse_numerical_converter = {item:key for key,item in numerical_converter.items()}
        reference_labels = np.array(self.reference.obs.loc[self.feature_tree_dict['labels'],'neuron'].map(numerical_converter).values)
        prediction = np.array(pd.Series(1*(np.mean(reference_labels[neighbors],axis=1)>0.5)).map(referse_numerical_converter))
        converter = dict(zip(cts,prediction))

        self.measured.obs['neuron'] = self.measured.obs['leiden'].map(converter)
        self.measured.obs['neuron_color'] = self.measured.obs['neuron'].map(self.neuron_color_converter)

        # self.update_user("Querying Reference tree")
        # neighbors,distances = self.feature_tree_dict['tree'].query(self.measured.layers['classification_space'],k=25)
        # self.update_user("Calculating Neuron Annotation")
        # numerical_converter = {'Neuron':0,'Non_Neuron':1}
        # referse_numerical_converter = {item:key for key,item in numerical_converter.items()}
        # reference_labels = np.array(self.reference.obs.loc[self.feature_tree_dict['labels'],'neuron'].map(numerical_converter).values)
        # prediction = np.array(pd.Series(1*(np.mean(reference_labels[neighbors],axis=1)>0.5)).map(referse_numerical_converter))
        # self.measured.obs['neuron'] = prediction
        # self.measured.obs['neuron_color'] = self.measured.obs['neuron'].map(self.neuron_color_converter)

        # Update Priors
        self.update_user("Updating Priors")
        prior_matrix = self.priors['matrix'].copy()
        neuron_idxs = [idx for idx,ct in enumerate(self.priors['columns']) if not 'NN' in ct]
        m = self.measured.obs['neuron']=='Non_Neuron'
        nn_priors = self.priors['matrix'][m,:].copy()
        nn_priors[:,neuron_idxs] = 0
        prior_matrix[m,:] = nn_priors

        non_neuron_idxs = [idx for idx,ct in enumerate(self.priors['columns']) if 'NN' in ct]
        m = self.measured.obs['neuron']=='Neuron'
        n_priors = self.priors['matrix'][m,:].copy()
        n_priors[:,non_neuron_idxs] = 0
        prior_matrix[m,:] = n_priors

        self.priors['matrix'] = prior_matrix.copy()
        self.visualize_measured('neuron_color','Neuron Annotation')
        self.visualize_layers('classification_space',measured_color='neuron_color',reference_color='neuron_color',reference_layer = 'classification_space')


    def supervised_harmonization(self):
        self.update_user("Performing Supervised Harmonization")
        
        self.update_user("Creating Dendrogram")
        X = np.array(self.reference.layers['classification_space'])
        y = np.array(self.reference.obs[self.ref_level])
        unique_labels = np.unique(y)
        averaged_data = {label: np.median(X[y == label],axis=0) for label in unique_labels}
        df = pd.DataFrame(averaged_data)
        correlations = df.corr()
        condensed_distance_matrix = squareform(np.abs(correlations-1), checks=False)
        self.Z = linkage(condensed_distance_matrix, method='ward')
        self.average_df = df.copy()
        clusters = fcluster(self.Z, t=self.Z[0,2], criterion='distance')
        column_clusters = {col: cluster for col, cluster in zip(self.average_df.columns, clusters)}

        dend = pd.DataFrame(index=self.average_df.columns,columns=range(self.Z.shape[0]))
        stop=False
        for n in range(self.Z.shape[0]):
            clusters = np.array(fcluster(self.Z, t=self.Z[-(n+1),2], criterion='distance'))
            if n==0:
                dend[n] = clusters
                column_clusters = {col: cluster for col, cluster in zip(self.average_df.columns, clusters)}
                previous_labels = pd.Series(y).map(column_clusters)
                continue
            clusters = clusters+dend[n-1].max()
            column_clusters = {col: cluster for col, cluster in zip(self.average_df.columns, clusters)}

            updated_labels = pd.Series(y).map(column_clusters)
            updated_label_counts = updated_labels.value_counts()
            previous_label_counts = previous_labels.value_counts()
            """ check for clusters with the same number of cells"""
            for idx in updated_label_counts.index:
                count = updated_label_counts[idx]
                for idx2 in previous_label_counts[previous_label_counts==count].index:
                    """ check if they are for the same base_types"""
                    if np.mean(np.array(dend[n-1]==idx2)==np.array(clusters==idx))==1:
                        clusters[clusters==idx] = idx2
            dend[n] = clusters
            column_clusters = {col: cluster for col, cluster in zip(self.average_df.columns, clusters)}
            previous_labels = pd.Series(y).map(column_clusters)
        """ Rename to have no gaps in names"""
        mapper = dict(zip(sorted(np.unique(dend)),range(len(np.unique(dend)))))
        for i in dend.columns:
            dend[i] = dend[i].map(mapper)
        self.dend = dend

        """ Remake Priors to use dend clusters """
        self.update_user("Remaking Priors")
        types = np.unique(np.array(self.dend).ravel())
        priors = np.zeros((self.measured.shape[0],types.shape[0])).astype(np.float32)
        updated_priors = {'columns':types,'indexes':np.array(self.measured.obs.index),'matrix':priors.astype(np.float32)}
        for i,cluster in enumerate(types):
            included_labels = np.array(self.dend.index)[(self.dend==cluster).max(axis=1)==1]
            updated_priors['matrix'][:,i] = np.sum(self.priors['matrix'][:,np.isin(self.priors['columns'],included_labels)],axis=1)
            # if not self.use_prior:
            #     updated_priors['matrix'][:,i] = np.ones_like(updated_priors['matrix'][:,i])
        self.dend_priors = updated_priors

        self.reference_features = X
        self.reference_labels_ref = y
        self.reference_cell_names = np.array([i.split('raise')[0] for i in self.reference.obs.index])

        self.measured_features = self.measured.layers['classification_space'].copy()
        self.measured_labels = np.zeros(self.measured_features.shape[0])

        self.measured_features = basicu.zscore_matching(self.reference_features,self.measured_features)
        self.measured.layers['zscored'] = self.measured_features.copy()
        self.update_user("Harmonizing")
        # set up empty likelihoods and posteriors to match priors
        self.likelihoods = {'columns':self.dend_priors['columns'],'indexes':np.array(self.measured.obs.index),'matrix':np.zeros_like(self.dend_priors['matrix'])}
        self.posteriors = {'columns':self.dend_priors['columns'],'indexes':np.array(self.measured.obs.index),'matrix':np.zeros_like(self.dend_priors['matrix'])}
        self.measured_features = self.measured.layers['classification_space'].copy()
        completed_clusters = []

        clusters = np.array(sorted(np.unique(np.array(self.dend).ravel())))
        if self.verbose:
            iterable = tqdm(clusters,desc='Harmonizing')
        else:
            iterable = clusters
        for cluster in iterable:
            if cluster in self.dend[self.dend.columns[-1]].unique():
                """ Reached the end of this branch"""
                continue
            mask = self.measured_labels==cluster
            self.likelihoods['matrix'][mask,:] = 0
            if np.sum(mask)==0:
                continue
            n = np.max(self.dend.columns[(self.dend==cluster).max(0)])
            if n==self.dend.columns[-1]:
                """ Reached the end of this branch"""
                continue
            mapper = dict(self.dend[n+1])
            next_clusters = self.dend[n+1][self.dend[n]==cluster].unique()
            self.reference_labels = np.array(pd.Series(self.reference_labels_ref).map(mapper))
            ref_m = np.isin(self.reference_labels, next_clusters) 
            self.model.fit(self.reference_features[ref_m,:],self.reference_labels[ref_m])

            likelihoods = self.model.predict_proba(self.measured_features[mask,:]).astype(np.float32)
            for idx,ct in enumerate(self.model.classes_):
                jidx = np.where(self.likelihoods['columns']==ct)[0][0]
                self.likelihoods['matrix'][mask,jidx] = likelihoods[:,idx]
            likelihoods = self.likelihoods['matrix'][mask,:].copy()
            likelihoods[:,np.isin(self.dend_priors['columns'],next_clusters)==False] = 0
            priors = self.dend_priors['matrix'][mask,:].copy()
            priors[:,np.isin(self.dend_priors['columns'],next_clusters)==False] = 0

            if self.likelihood_only:
                posteriors = likelihoods
            elif self.prior_only:
                posteriors = priors
            else:
                posteriors = likelihoods*priors
                posteriors[posteriors.max(1)==0,:] = likelihoods[posteriors.max(1)==0,:].copy()

            labels = self.dend_priors['columns'][np.argmax(posteriors,axis=1)]
            labels[posteriors.max(1)==0] = -1
            self.measured_labels[mask] = labels
            for cluster in np.unique(self.measured_labels[mask]):
                m = self.measured_labels==cluster
                ref_m = self.reference_labels==cluster
                if np.sum(m)>0:
                    if cluster ==-1:
                        self.update_user(f"{np.sum(m)} cells 0 posterior")
                        continue
                    self.measured_features[m,:] = basicu.zscore_matching(self.reference_features[ref_m,:],self.measured_features[m,:])
            gc.collect()

        self.update_user(f"Mapping labels to {self.ref_level}")
        mapper = dict(self.dend[self.dend.columns[-1]])
        reverse_mapper = {v:k for k,v in mapper.items()}
        self.measured_labels = np.array(pd.Series(self.measured_labels).map(reverse_mapper))
        self.measured.obs[self.ref_level] = self.measured_labels
        self.measured.obs[f"{self.ref_level}_color"] = self.measured.obs[self.ref_level].map(self.ref_level_color_converter)
        self.visualize_measured(f"{self.ref_level}_color",'Supervised Annotation')
        self.measured.layers['harmonized_classification_space'] = self.measured_features.copy()
        self.measured.layers['harmonized'] = (self.measured_features.copy()*self.ref_stds)+self.ref_medians
        
        self.visualize_layers('harmonized',measured_color='leiden_color',reference_color=f"{self.ref_level}_color",reference_layer = 'normalized')
        self.visualize_layers('harmonized',measured_color=f"{self.ref_level}_color",reference_color=f"{self.ref_level}_color",reference_layer = 'normalized')

    def determine_neighbors(self):
        self.update_user("Determining Neighbors")
        
        self.update_user("Querying Reference tree")
        neighbors,distances = self.feature_tree_dict['tree'].query(self.measured.layers['harmonized_classification_space'],k=15)
        X = np.zeros_like(self.measured.layers['harmonized_classification_space'])
        for i in range(neighbors.shape[1]):
            self.measured.obs[f"reference_neighbor_{i}"] = np.array(self.feature_tree_dict['labels'][neighbors[:,i]])
            X = X+np.array(self.reference.layers['raw'][neighbors[:,i],:])
        X = X/neighbors.shape[1]
        self.measured.layers['imputed'] = X.copy()

        self.visualize_layers('imputed',measured_color=f"{self.ref_level}_color",reference_color=f"{self.ref_level}_color",reference_layer = 'raw')

    def run(self):
        self.unsupervised_clustering()
        self.calculate_spatial_priors()
        self.load_reference()
        self.unsupervised_neuron_annotation()
        self.supervised_harmonization()
        self.determine_neighbors()
        return self.measured

    def visualize_measured(self,color,title):
        if self.visualize:
            adata = self.measured.copy()
            x = adata.obs['ccf_z']
            y = adata.obs['ccf_y']
            c = np.array(adata.obs[color])
            fig,ax  = plt.subplots(1,1,figsize=[5,5])
            ax.scatter(x,y,s=0.1,c=c,marker='.')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.axis('off')
            plt.show()

    def visualize_layers(self,layer,measured_color='leiden_color',reference_color='',reference_layer = ''):
        if self.visualize:
            adata = self.measured.copy()
            ref_adata = self.reference.copy()
            if reference_color == '':
                if measured_color in ref_adata.obs.columns:
                    reference_color = measured_color
                else:
                    reference_color = f"{self.ref_level}_color"
            if reference_layer == '':
                if layer in ref_adata.layers.keys():
                    reference_layer = layer
                else:
                    reference_layer = 'classification_space'

            reference_features = self.reference.layers[reference_layer].copy()
            measured_features = self.measured.layers[layer].copy()
            for bit1 in np.arange(0,self.reference.shape[1],2):
                bit2 = bit1+1
                
                percentile_min = 0.1
                percentile_max = 99.9
                # Calculate percentiles for reference data
                ref_x_min, ref_x_max = np.percentile(reference_features[:, bit1], [percentile_min, percentile_max])
                ref_y_min, ref_y_max = np.percentile(reference_features[:, bit2], [percentile_min, percentile_max])

                # Calculate percentiles for measured data
                meas_x_min, meas_x_max = np.percentile(measured_features[:, bit1], [percentile_min, percentile_max])
                meas_y_min, meas_y_max = np.percentile(measured_features[:, bit2], [percentile_min, percentile_max])

                # meas_x_min = np.max([meas_x_min,1])
                # meas_y_min = np.max([meas_y_min,1])
                meas_x_min = ref_x_min
                meas_y_min = ref_y_min
                meas_x_max = ref_x_max
                meas_y_max = ref_y_max

                # Clip the values and count how many points are above the limits
                ref_x_clipped = np.clip(reference_features[:, bit1], ref_x_min, ref_x_max)
                ref_y_clipped = np.clip(reference_features[:, bit2], ref_y_min, ref_y_max)
                meas_x_clipped = np.clip(measured_features[:, bit1], meas_x_min, meas_x_max)
                meas_y_clipped = np.clip(measured_features[:, bit2], meas_y_min, meas_y_max)

                ref_order = np.argsort(reference_features[:, bit1])
                meas_order = np.argsort(measured_features[:, bit1])
                ref_order = np.random.choice(ref_order, meas_order.shape[0])
                np.random.shuffle(ref_order)
                np.random.shuffle(meas_order)
                
                x_bins = np.linspace(ref_x_min, ref_x_max, 100)
                y_bins = np.linspace(ref_y_min, ref_y_max, 100)

                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                axs = axs.ravel()

                # Calculate density for reference data
                ref_density, ref_x_edges, ref_y_edges = np.histogram2d(ref_x_clipped[ref_order], ref_y_clipped[ref_order], bins=(x_bins, y_bins))
                # vmin, vmax = np.percentile(ref_density[ref_density > 0], [1, 99])
                # Plot density for reference data
                ref_density = np.flipud(ref_density.T)
                ref_density = np.log10(ref_density + 1)
                vmin, vmax = np.percentile(ref_density, [1, 99])
                im = axs[0].imshow(ref_density, cmap='jet', vmin=vmin, vmax=vmax)#, extent=[ref_x_min, ref_x_max, ref_y_min, ref_y_max])
                cbar = plt.colorbar(im, ax=axs[0])
                cbar.set_label('log10(Density)')
                axs[0].set_title('Reference Density')
                axs[0].axis('off')
                # axs[0].set_xlim(ref_x_min, ref_x_max)
                # axs[0].set_ylim(ref_y_min, ref_y_max)

                # Calculate density for measured data
                meas_density, meas_x_edges, meas_y_edges = np.histogram2d(meas_x_clipped[meas_order], meas_y_clipped[meas_order], bins=(x_bins, y_bins))
                # vmin, vmax = np.percentile(meas_density[meas_density > 0], [1, 99])
                # Plot density for measured data
                meas_density = np.flipud(meas_density.T)
                meas_density = np.log10(meas_density + 1)
                vmin, vmax = np.percentile(meas_density, [1, 99])
                im = axs[1].imshow(meas_density, cmap='jet', vmin=vmin, vmax=vmax)#, extent=[meas_x_min, meas_x_max, meas_y_min, meas_y_max])
                cbar = plt.colorbar(im, ax=axs[1])
                cbar.set_label('log10(Density)')
                axs[1].set_title('Measured Density')
                axs[1].axis('off')
                # axs[1].set_xlim(meas_x_min, meas_x_max)
                # axs[1].set_ylim(meas_y_min, meas_y_max)

                # Plot reference data
                axs[2].scatter(ref_x_clipped[ref_order], ref_y_clipped[ref_order], s=1, c=np.array(ref_adata.obs[reference_color])[ref_order], marker='.')
                axs[2].set_xlim(ref_x_min, ref_x_max)
                axs[2].set_ylim(ref_y_min, ref_y_max)
                axs[2].set_title('Reference')
                axs[2].set_xlabel(f"{self.reference.var.index[bit1]}")
                axs[2].set_ylabel(f"{self.reference.var.index[bit2]}")

                # Plot measured data
                axs[3].scatter(meas_x_clipped[meas_order], meas_y_clipped[meas_order], s=1, c=np.array(adata.obs[measured_color])[meas_order], marker='.')
                axs[3].set_xlim(meas_x_min, meas_x_max)
                axs[3].set_ylim(meas_y_min, meas_y_max)
                axs[3].set_title('Measured')

                plt.tight_layout()
                plt.show()
