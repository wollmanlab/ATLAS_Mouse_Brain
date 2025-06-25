import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import torch
import json
import multiprocessing
import anndata

# Add the dredFISH path
sys.path.append('/home/rwollman/MyProjects/AH/Repos/dredFISH')
from dredFISH.Analysis.TissueGraph import *
from dredFISH.Utils.imageu import gaussian_smoothing_with_nans, zoom_with_nans


def process_dataset(d, basepath):
    """Process a single dataset to extract cell type and spatial information."""
    print(f"loading {d}")
    path = os.path.join(basepath, d)
    filename = os.path.join(path, 'Layer', 'cell_layer.h5ad')
    adata = anndata.read_h5ad(filename)

    Type = adata.obs['subclass_id']
    XYZ = adata.obs[['ccf_z', 'ccf_y', 'ccf_x']].to_numpy()
    XYZT = np.hstack((np.array(XYZ), np.array(Type)[:, np.newaxis]))
    return d, Type, XYZ, XYZT


def create_5d_tensor(basepath='/scratchdata1/MouseBrainAtlases_V6/',
                     tax_basepath="/scratchdata1/MouseBrainAtlases_V6/Taxonomies",
                     minZ=2.5, maxZ=9.65, dx=0.1, dy=0.1, dz_zoom_factor=3, sig=1.5,
                     name='Allen_2p5_9p5_sig1p5_dz3x_2',
                     include_allen=True,
                     save_outputs=True):
    """
    Create 5D tensor representing cell type distributions across spatial coordinates.
    
    Parameters:
    -----------
    basepath : str
        Base path to the datasets
    tax_basepath : str
        Base path to the taxonomies
    minZ, maxZ : float
        Z-coordinate range
    dx, dy : float
        Spatial resolution in x and y dimensions
    dz_zoom_factor : int
        Zoom factor for z dimension
    sig : float
        Gaussian smoothing parameter
    name : str
        Configuration name for output files
    include_allen : bool
        Whether to include Allen dataset
    save_outputs : bool
        Whether to save output files
    
    Returns:
    --------
    dict : Dictionary containing the created tensors and metadata
    """
    
    # Load taxonomies
    cell_tax_to_build = ['class', 'subclass', 'supertype']
    Taxonomies = dict()
    for tx in cell_tax_to_build:
        Taxonomies[tx] = Taxonomy(tx, basepath=tax_basepath)
        Taxonomies[tx].load()
    
    # Create configuration
    dens_config = {
        'minZ': minZ,
        'maxZ': maxZ,
        'dx': dx,
        'dy': dy,
        'dz_zoom_factor': dz_zoom_factor,
        'sig': sig,
        'name': name
    }
    
    # Save configuration
    if save_outputs:
        config_filename = f"dens_config_{name}.json"
        with open(config_filename, 'w') as config_file:
            json.dump(dens_config, config_file, indent=4)
    
    # Create bins
    dz = dx * dz_zoom_factor
    x_bins = np.arange(0, 11.4 + dx, dx)
    y_bins = np.arange(0, 8 + dy, dy)
    z_bins = np.arange(minZ, maxZ + dz, dz)
    z_bins_zm = np.arange(minZ, maxZ + dx, dx)
    
    x_bins_cntr = (x_bins[:-1] + x_bins[1:]) / 2
    y_bins_cntr = (y_bins[:-1] + y_bins[1:]) / 2
    z_bins_cntr = (z_bins[:-1] + z_bins[1:]) / 2
    z_bins_zm_cntr = (z_bins_zm[:-1] + z_bins_zm[1:]) / 2
    
    bins = [x_bins, y_bins, z_bins]
    n = [len(b) - 1 for b in bins]
    Ntypes = Taxonomies['subclass'].N
    n.append(Ntypes)
    n_zm = n.copy()
    n_zm[2] = len(z_bins_zm_cntr)
    
    print(f"Tensor dimensions: {n}")
    print(f"Zoomed tensor dimensions: {n_zm}")
    
    # Define datasets
    male_datasets = ['MMSM01', 'WTM01', 'WTM04', 'WTM07', 'WTM11']
    female_datasets = ['MMSF01', 'WTF01', 'WTF04', 'WTF06', 'WTF11']
    asd_datasets = ['ASDM02', 'ASDM12', 'ASDM13', 'ASDM14', 'ASDM15']
    datasets = male_datasets + female_datasets + asd_datasets
    
    female_datasets_ix = np.array([datasets.index(d) for d in female_datasets if d in datasets])
    male_datasets_ix = np.array([datasets.index(d) for d in male_datasets if d in datasets])
    asd_datasets_ix = np.array([datasets.index(d) for d in asd_datasets if d in datasets])
    
    # Load or process dataset data
    XYZ = dict()
    Type = dict()
    XYZT = dict()
    all_xyzt_file = 'All_XYZT.npz'
    
    if os.path.exists(all_xyzt_file):
        print(f"Loading existing {all_xyzt_file}")
        loaded_data = np.load(all_xyzt_file)
        for d in datasets:
            if d in loaded_data:
                XYZT[d] = loaded_data[d]
                XYZ[d] = XYZT[d][:, :3]
                Type[d] = XYZT[d][:, 3]
            else:
                print(f"Warning: {d} not found in {all_xyzt_file}")
    else:
        print(f"{all_xyzt_file} not found. Processing datasets...")
        with multiprocessing.Pool() as pool:
            results = pool.map(lambda d: process_dataset(d, basepath), datasets)
        
        for d, type_data, xyz_data, xyzt_data in results:
            Type[d] = type_data
            XYZ[d] = xyz_data
            XYZT[d] = xyzt_data
        
        if save_outputs:
            np.savez(all_xyzt_file, **XYZT)
    
    # Process Allen dataset if requested
    if include_allen:
        allen_xyzt_file = 'Allen_xyzt.npz'
        if os.path.exists(allen_xyzt_file):
            print(f"Loading existing {allen_xyzt_file}")
            loaded_data = np.load(allen_xyzt_file)
            XYZT['Allen'] = loaded_data['Allen']
            XYZ['Allen'] = XYZT['Allen'][:, :3]
            Type['Allen'] = XYZT['Allen'][:, 3]
        else:
            print("Processing Allen dataset...")
            d, type_data, xyz_data, xyzt_data = process_dataset('Allen', basepath)
            Type[d] = type_data
            XYZ[d] = xyz_data
            XYZT[d] = xyzt_data
            if save_outputs:
                np.savez(allen_xyzt_file, Allen=xyzt_data)
    
    # Create 5D tensor for all datasets
    type_dist_4D = np.zeros(np.hstack([len(datasets), n_zm]).astype(int))
    voxel_cells_num = np.zeros(np.hstack([len(datasets), n[:-1]]).astype(int))
    
    print("Overall cells counting")
    for i, d in enumerate(datasets):
        voxel_cells_num[i, :, :, :], edges = np.histogramdd(XYZ[d], bins=bins)
    
    avg_cells_per_voxel = np.nanmean(voxel_cells_num, axis=0)
    voxel_corr_factor = avg_cells_per_voxel / voxel_cells_num
    
    print("Per cell type counting")
    for i, d in enumerate(datasets):
        print(d)
        for typ in range(Ntypes):
            # estimate spatial abundance 
            dens, edges = np.histogramdd(XYZ[d][Type[d] == typ, :], bins=bins)
            dens[voxel_cells_num[i, :, :, :] < 0.5 * avg_cells_per_voxel] = np.nan
            dens *= voxel_corr_factor[i, :, :, :]
            dens = zoom_with_nans(dens, (1, 1, dz_zoom_factor))
            dens = gaussian_smoothing_with_nans(dens, sig)
            type_dist_4D[i, :, :, :, typ] = dens
    
    # Convert to torch tensors
    voxel_cells_num = torch.Tensor(voxel_cells_num)
    type_dist_4D = torch.Tensor(type_dist_4D)
    
    # Create Allen tensor if requested
    allen_type_dist_4D = None
    allen_voxel_cells_num = None
    
    if include_allen:
        allen_type_dist_4D = np.zeros(n_zm).astype(int)
        allen_voxel_cells_num = np.zeros(n[:-1]).astype(int)
        
        print("Per cell type counting for Allen")
        for i, d in enumerate(['Allen']):
            print(d)
            for typ in range(Ntypes):
                # estimate spatial abundance 
                dens, edges = np.histogramdd(XYZ[d][Type[d] == typ, :], bins=bins)
                dens = zoom_with_nans(dens, (1, 1, dz_zoom_factor))
                dens = gaussian_smoothing_with_nans(dens, sig)
                allen_type_dist_4D[:, :, :, typ] = dens
        
        allen_voxel_cells_num = torch.Tensor(allen_voxel_cells_num)
        allen_type_dist_4D = torch.Tensor(allen_type_dist_4D)
    
    # Save outputs if requested
    if save_outputs:
        torch.save(voxel_cells_num, f"voxel_cells_num_{dens_config['name']}.pt")
        torch.save(type_dist_4D, f"type_dist_4D_{dens_config['name']}.pt")
        
        if include_allen:
            torch.save(allen_voxel_cells_num, f"allen_voxel_cells_num_{dens_config['name']}.pt")
            torch.save(allen_type_dist_4D, f"allen_type_dist_4D_{dens_config['name']}.pt")
    
    # Return results
    results = {
        'type_dist_4D': type_dist_4D,
        'voxel_cells_num': voxel_cells_num,
        'allen_type_dist_4D': allen_type_dist_4D,
        'allen_voxel_cells_num': allen_voxel_cells_num,
        'dens_config': dens_config,
        'bins': bins,
        'n': n,
        'n_zm': n_zm,
        'datasets': datasets,
        'female_datasets_ix': female_datasets_ix,
        'male_datasets_ix': male_datasets_ix,
        'asd_datasets_ix': asd_datasets_ix,
        'XYZ': XYZ,
        'Type': Type,
        'XYZT': XYZT
    }
    
    return results


def permute_hemis(cond1_hemi_ix, cond2_hemi_ix, fullperm=True):
    """
    Permute hemisphere indices between two conditions.
    
    Parameters:
    -----------
    cond1_hemi_ix : np.array
        Hemisphere indices for condition 1
    cond2_hemi_ix : np.array
        Hemisphere indices for condition 2
    fullperm : bool
        If True, do full permutation. If False, do partial permutation.
    
    Returns:
    --------
    list : [permuted_cond1_ix, permuted_cond2_ix]
    """
    if fullperm:
        perm = np.hstack((cond1_hemi_ix, cond2_hemi_ix))
        np.random.shuffle(perm)
        perm = [perm[:len(cond1_hemi_ix)], perm[len(cond1_hemi_ix):]]
    else:
        selected_cond1_ix = np.random.choice(cond1_hemi_ix, len(cond1_hemi_ix) // 2, replace=False)
        selected_cond2_ix = np.random.choice(cond2_hemi_ix, len(cond2_hemi_ix) // 2, replace=False)

        remaining_cond1_ix = np.setdiff1d(cond1_hemi_ix, selected_cond1_ix)
        remaining_cond2_ix = np.setdiff1d(cond2_hemi_ix, selected_cond2_ix)

        perm = [np.concatenate((selected_cond1_ix, selected_cond2_ix)),
                np.concatenate((remaining_cond1_ix, remaining_cond2_ix))]
    return perm


def calculate_entropy_residual(grouping, type_dist_hemi_single, type_dist_hemi_not_nans, res_limits=None):
    """
    Calculate entropy of residual distribution between two groups.
    
    This is the core function that implements the entropy residual analysis.
    
    Parameters:
    -----------
    grouping : list
        List of two arrays containing indices for each group
    type_dist_hemi_single : torch.Tensor
        Single type distribution tensor (hemisphere data)
    type_dist_hemi_not_nans : torch.Tensor
        Boolean tensor indicating valid (non-NaN) values
    res_limits : tuple, optional
        (min_res, max_res) for histogram calculation. If None, calculated from data.
    
    Returns:
    --------
    torch.Tensor or tuple : 
        If res_limits is None: (entropy_score, res_limits)
        If res_limits is provided: entropy_score
    """
    stk_grp1_sum = type_dist_hemi_single[grouping[0]].sum(dim=0)
    if stk_grp1_sum.max() == 0:
        if res_limits is None:
            return torch.nan, torch.nan
        else:
            return torch.nan, torch.nan

    stk_grp2_sum = type_dist_hemi_single[grouping[1]].sum(dim=0)
    if stk_grp2_sum.max() == 0:
        if res_limits is None:
            return torch.nan, torch.nan
        else:
            return torch.nan, torch.nan

    grp1_valid_count = type_dist_hemi_not_nans[grouping[0]].sum(dim=0)
    grp2_valid_count = type_dist_hemi_not_nans[grouping[1]].sum(dim=0)

    stk_grp1_avg = stk_grp1_sum / grp1_valid_count
    stk_grp1_avg[torch.isnan(stk_grp1_avg)] = 0.0
    stk_grp2_avg = stk_grp2_sum / grp2_valid_count
    stk_grp2_avg[torch.isnan(stk_grp2_avg)] = 0.0

    # Calculate residual distribution
    p = stk_grp1_avg / stk_grp1_avg.sum()
    q = stk_grp2_avg / stk_grp2_avg.sum()
    res = p - q

    if res_limits is None:
        return_res_limits = True
        min_res = res.min()
        max_res = res.max()
        res_limits = (min_res, max_res)
    else:
        min_res = res_limits[0]
        max_res = res_limits[1]
        return_res_limits = False

    # Calculate histogram
    hist = torch.histc(res, bins=256, min=min_res, max=max_res)
    
    # Normalize to get probability distribution
    prob_dist = hist / torch.sum(hist)
    
    # Calculate entropy
    entropy = -torch.sum(torch.where(prob_dist > 0, 
                                   prob_dist * torch.log2(prob_dist), 
                                   torch.tensor(0.0, device=prob_dist.device)))
    
    if return_res_limits:
        return entropy, res_limits
    else:
        return entropy


def run_permutation_test(type_dist_4D, male_hemi_ix, female_hemi_ix, asd_hemi_ix, 
                        min_iter=900, extra_iter=100, max_iter=100000, 
                        conditions=['dens_fm', 'dens_asd'], use_cuda=True):
    """
    Run permutation test for type distribution differences.
    
    Parameters:
    -----------
    type_dist_4D : torch.Tensor
        5D tensor of type distributions
    male_hemi_ix : np.array
        Male hemisphere indices
    female_hemi_ix : np.array
        Female hemisphere indices  
    asd_hemi_ix : np.array
        ASD hemisphere indices
    min_iter : int
        Minimum iterations for burn-in
    extra_iter : int
        Extra iterations per adaptive step
    max_iter : int
        Maximum total iterations
    conditions : list
        List of conditions to test
    use_cuda : bool
        Whether to use CUDA for computations
    
    Returns:
    --------
    dict : Dictionary containing scores, null distributions, and p-values
    """
    Ntypes = type_dist_4D.shape[-1]
    types_to_test = np.arange(Ntypes)
    
    # Create hemisphere tensor (left + flipped right)
    type_dist_hemi_torch = torch.cat((type_dist_4D[:, :57, :, :], 
                                     torch.flip(type_dist_4D[:, 57:, :, :], dims=[1])), dim=0)
    
    if use_cuda:
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Initialize results dictionaries
    scr = {}
    scr_null = {}
    pval = {}
    non_male_hemi_ix = {}
    
    for c in conditions:
        scr[c] = np.full((len(types_to_test), 1), np.nan)
        scr_null[c] = np.full((len(types_to_test), max_iter), np.nan)
        pval[c] = np.zeros(len(types_to_test))
        if 'fm' in c:
            non_male_hemi_ix[c] = female_hemi_ix
        else:
            non_male_hemi_ix[c] = asd_hemi_ix
    
    # Run permutation tests for each type
    for it, typ in enumerate(types_to_test):
        print(f"Type: {typ}")
        type_dist_hemi_single = type_dist_hemi_torch[:, :, :, :, typ].to(device)
        type_dist_hemi_not_nans = (~torch.isnan(type_dist_hemi_single)).float()
        type_dist_hemi_single[torch.isnan(type_dist_hemi_single)] = 0.0
        
        for c in conditions:
            i = 0
            grp = [male_hemi_ix, non_male_hemi_ix[c]]
            scr[c][it, 0], res_limits = calculate_entropy_residual(grp, type_dist_hemi_single, 
                                                                  type_dist_hemi_not_nans)
            
            if np.isnan(scr[c][it]):
                pval[c][it] = np.nan
                continue
            
            # Initial burn-in iterations
            for ii in range(min_iter):
                rnd_grp = permute_hemis(male_hemi_ix, non_male_hemi_ix[c])
                scr_null[c][it, i] = calculate_entropy_residual(rnd_grp, type_dist_hemi_single, 
                                                              type_dist_hemi_not_nans, res_limits=res_limits)
                i += 1
            
            # Adaptive iterations
            while ~np.isnan(pval[c][it]) and pval[c][it] < 5/(i+1) and i < max_iter:
                ii = 0
                while ii < extra_iter:
                    rnd_grp = permute_hemis(male_hemi_ix, non_male_hemi_ix[c])
                    scr_null[c][it, i] = calculate_entropy_residual(rnd_grp, type_dist_hemi_single, 
                                                                  type_dist_hemi_not_nans, res_limits=res_limits)
                    ii += 1
                    i += 1
                pval[c][it] = 1 - np.mean(scr[c][it] > scr_null[c][it, :i])
                if i % 1000 == 0:
                    print(f"{i} iters | pval = {pval[c][it]:.5f}")
            
            print(f"after {i} iters: Cond={c} | pval = {pval[c][it]:.5f} | "
                  f"scr={scr[c][it, 0]:.4f} | max of null: {np.nanmax(scr_null[c][it, :]):.4f}")
    
    return {
        'scr': scr,
        'scr_null': scr_null,
        'pval': pval,
        'conditions': conditions,
        'types_to_test': types_to_test
    }


def calculate_multiple_testing_correction(pval, avg_M, avg_F, avg_ASD, conditions=['dens_fm', 'dens_asd']):
    """
    Calculate multiple testing corrections for permutation test results.
    
    Parameters:
    -----------
    pval : dict
        Dictionary of p-values for each condition
    avg_M, avg_F, avg_ASD : np.array
        Average cell counts for each group
    conditions : list
        List of conditions
    
    Returns:
    --------
    dict : Dictionary containing corrected p-values and significant hits
    """
    from statsmodels.stats.multitest import multipletests
    
    min_count_FM = np.minimum(avg_M, avg_F)
    min_count_ASD = np.minimum(avg_M, avg_ASD)
    hits = {}
    corrected_pvals = {}
    
    for c in conditions:
        p = pval[c].copy()
        if 'fm' in c:
            ix = np.flatnonzero((np.isfinite(p)) & (min_count_FM > 100) & (p < 1))
        else:
            ix = np.flatnonzero((np.isfinite(p)) & (min_count_ASD > 100) & (p < 1))
        
        p = p[ix]
        _, corr_p, _, _ = multipletests(p, method='fdr_bh')
        corrected_pvals[c] = np.full(len(pval[c]), 0.5)
        corrected_pvals[c][ix] = corr_p
        hits[c] = np.flatnonzero(corrected_pvals[c] <= 0.05)
    
    return {
        'corrected_pvals': corrected_pvals,
        'hits': hits,
        'min_count_FM': min_count_FM,
        'min_count_ASD': min_count_ASD
    }


def torch_nanmean(T, axis=None, keepdim=False):
    """
    Calculate mean of tensor while handling NaN values.
    
    Parameters:
    -----------
    T : torch.Tensor
        Input tensor
    axis : int or tuple, optional
        Axis along which to calculate mean
    keepdim : bool
        Whether to keep dimensions
    
    Returns:
    --------
    torch.Tensor : Mean values
    """
    Tmsk = torch.isnan(T)
    Tnanzeroed = T.masked_fill(Tmsk, 0)
    if axis is None:
        Tsm = Tnanzeroed.sum() / (~Tmsk).sum()
    else:
        Tsm = torch.sum(Tnanzeroed, dim=axis, keepdim=keepdim) / torch.sum(~Tmsk, dim=axis, keepdim=keepdim)
    return Tsm


def torch_nansum(T, axis=None, keepdim=False):
    """
    Calculate sum of tensor while handling NaN values.
    
    Parameters:
    -----------
    T : torch.Tensor
        Input tensor
    axis : int or tuple, optional
        Axis along which to calculate sum
    keepdim : bool
        Whether to keep dimensions
    
    Returns:
    --------
    torch.Tensor : Sum values
    """
    Tmsk = torch.isnan(T)
    Tnanzeroed = T.masked_fill(Tmsk, 0)
    if axis is None:
        Tsm = Tnanzeroed.sum()
    else:
        Tsm = torch.sum(Tnanzeroed, dim=axis, keepdim=keepdim)
    return Tsm


def gpu_calc_diff_map_apt(gpu_animal_pos_type_no_nans, gpu_animal_pos_type_is_nans, Pix, G_ix1, G_ix2):
    """
    Calculate difference map using correlation analysis on GPU.
    
    Parameters:
    -----------
    gpu_animal_pos_type_no_nans : torch.Tensor
        GPU tensor of animal position type data (NaN values replaced with 0)
    gpu_animal_pos_type_is_nans : torch.Tensor
        GPU tensor indicating which values were originally NaN
    Pix : torch.Tensor
        Boolean tensor indicating which voxels to analyze
    G_ix1, G_ix2 : torch.Tensor
        Group indices for comparison
    
    Returns:
    --------
    torch.Tensor : Difference map (1 - correlation)
    """
    X = gpu_animal_pos_type_no_nans[G_ix1][:, Pix, :].sum(axis=0)
    X_count = (~gpu_animal_pos_type_is_nans[G_ix1][:, Pix, :]).float().sum(axis=0)
    X.div_(X_count)  # In-place division

    Y = gpu_animal_pos_type_no_nans[G_ix2][:, Pix, :].sum(axis=0)
    Y_count = (~gpu_animal_pos_type_is_nans[G_ix2][:, Pix, :]).float().sum(axis=0)
    Y.div_(Y_count)  # In-place division

    X_mean = torch.mean(X, dim=1, keepdim=True)
    Y_mean = torch.mean(Y, dim=1, keepdim=True)

    # Subtract means
    X.sub_(X_mean)  # In-place subtraction
    Y.sub_(Y_mean)

    # Calculate the numerator of the correlation coefficient
    numerator = torch.sum(X * Y, dim=1)

    # Calculate the denominator of the correlation coefficient
    X_std = torch.sqrt((X**2).sum(dim=1))
    Y_std = torch.sqrt((Y**2).sum(dim=1))
    denominator = X_std * Y_std

    # Calculate the row-wise correlation coefficients
    row_correlations = numerator / denominator

    diff_map = torch.tensor(1.0, device='cuda') - row_correlations
    diff_map = diff_map.cpu()
    
    return diff_map


def calc_diff_map_apt(Pix, G_ix1, G_ix2, animal_pos_type, min_cell_dens=0, return_rows=False):
    """
    Calculate difference map using correlation analysis on CPU.
    
    Parameters:
    -----------
    Pix : torch.Tensor
        Boolean tensor indicating which voxels to analyze
    G_ix1, G_ix2 : torch.Tensor or np.array
        Group indices for comparison
    animal_pos_type : torch.Tensor
        Animal position type data
    min_cell_dens : float
        Minimum cell density threshold
    return_rows : bool
        Whether to return individual group data
    
    Returns:
    --------
    torch.Tensor or tuple : Difference map and optionally group data
    """
    X = torch_nanmean(animal_pos_type[G_ix1][:, Pix, :], axis=0)
    Y = torch_nanmean(animal_pos_type[G_ix2][:, Pix, :], axis=0)
    
    # Apply minimum cell density filter
    X[(X < min_cell_dens) | (Y < min_cell_dens)] = torch.nan
    Y[(X < min_cell_dens) | (Y < min_cell_dens)] = torch.nan
    
    X_mean = torch_nanmean(X, axis=1, keepdim=True)
    Y_mean = torch_nanmean(Y, axis=1, keepdim=True)

    # Subtract means
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Calculate the numerator of the correlation coefficient
    numerator = torch_nansum(X_centered * Y_centered, axis=1)

    # Calculate the denominator of the correlation coefficient
    X_std = torch.sqrt(torch_nansum(X_centered**2, axis=1))
    Y_std = torch.sqrt(torch_nansum(Y_centered**2, axis=1))
    denominator = X_std * Y_std

    # Calculate the row-wise correlation coefficients
    row_correlations = numerator / denominator

    diff_map = 1 - row_correlations
    
    if return_rows:
        return diff_map, X, Y
    else:
        return diff_map


def calc_comp_pvals(animal_pos_type, Vix, Tix, cond1_hemi_ix, cond2_hemi_ix, 
                   save_file_name, rnd_iter=100, rnd_per_iter=100, use_cuda=True):
    """
    Calculate comparison p-values using permutation testing.
    
    Parameters:
    -----------
    animal_pos_type : torch.Tensor
        Animal position type data
    Vix : torch.Tensor
        Boolean tensor indicating valid voxels
    Tix : torch.Tensor
        Boolean tensor indicating valid types
    cond1_hemi_ix, cond2_hemi_ix : torch.Tensor or np.array
        Group indices for comparison
    save_file_name : str
        File name to save results
    rnd_iter : int
        Number of random iterations
    rnd_per_iter : int
        Number of permutations per iteration
    use_cuda : bool
        Whether to use CUDA for computations
    
    Returns:
    --------
    np.array : P-values for each voxel
    """
    if use_cuda:
        # Copy data to GPU
        gpu_animal_pos_type_no_nans = animal_pos_type.cuda()
        gpu_animal_pos_type_is_nans = torch.isnan(animal_pos_type).cuda()
        Pix = torch.ones(torch.sum(Vix), dtype=bool).cuda()
        gpu_cond1_hemi_ix = torch.tensor(cond1_hemi_ix, device='cuda')
        gpu_cond2_hemi_ix = torch.tensor(cond2_hemi_ix, device='cuda')
        
        # Calculate initial difference map
        diff_map = torch.full((len(Vix),), torch.nan)
        diff_map[Vix] = gpu_calc_diff_map_apt(gpu_animal_pos_type_no_nans, gpu_animal_pos_type_is_nans, 
                                             Pix, gpu_cond1_hemi_ix, gpu_cond2_hemi_ix)
        
        pvals = torch.zeros(len(Vix))
        diff_map_apt_null = torch.full((torch.sum(Vix), rnd_iter * rnd_per_iter), torch.nan)
        
        for i in range(rnd_iter):
            Pix = Pix & (pvals[Vix] < 0.05 * np.exp(-5 * (i * rnd_per_iter) / 10000)).cuda()
            Pix_indices = torch.nonzero(Pix).squeeze()
            print(f"{i}: analyzing {Pix.float().mean() * 100:.3f}% of voxels")
            
            for ii in range(rnd_per_iter):
                perm = permute_hemis(gpu_cond1_hemi_ix, gpu_cond2_hemi_ix)
                df = gpu_calc_diff_map_apt(gpu_animal_pos_type_no_nans, gpu_animal_pos_type_is_nans, 
                                          Pix, perm[0], perm[1])
                diff_map_apt_null[Pix_indices, i * rnd_per_iter + ii] = df
            
            pvals[Vix] = 1 - torch_nansum(diff_map[Vix, None] > diff_map_apt_null, axis=1) / torch_nansum(torch.isfinite(diff_map_apt_null), axis=1)
            np.save(save_file_name, pvals.numpy())
        
        pvals = pvals.numpy()
        pvals[pvals == 0] = 1 / (rnd_iter * rnd_per_iter)
        np.save(save_file_name, pvals)
        
        return pvals
    
    else:
        # CPU implementation (simplified version)
        print("CPU implementation not fully implemented. Use use_cuda=True for full functionality.")
        return None


def clear_cuda():
    """
    Clear CUDA memory and cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        for var_name in list(globals()):
            var = globals()[var_name]
            if isinstance(var, torch.Tensor) and var.is_cuda:
                del globals()[var_name]
        
        import gc
        gc.collect()
        
        print("All GPU variables have been deleted and cache cleared.")
    else:
        print("CUDA is not available. No GPU variables to delete.")

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        
        print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"Cached GPU memory: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")


def prepare_region_analysis_data(type_dist_4D, female_hemi_ix, male_hemi_ix, asd_hemi_ix, 
                                min_voxel_density=10, min_type_density=1000):
    """
    Prepare data for region-based analysis.
    
    Parameters:
    -----------
    type_dist_4D : torch.Tensor
        5D tensor of type distributions
    female_hemi_ix, male_hemi_ix, asd_hemi_ix : np.array
        Hemisphere indices for each group
    min_voxel_density : float
        Minimum voxel density threshold
    min_type_density : float
        Minimum type density threshold
    
    Returns:
    --------
    dict : Dictionary containing prepared data for analysis
    """
    # Create hemisphere tensor (left + flipped right)
    type_dist_hemi_torch = torch.cat((type_dist_4D[:, :57, :, :], 
                                     torch.flip(type_dist_4D[:, 57:, :, :], dims=[1])), dim=0)
    
    # Reshape to (animals, voxels, types)
    animal_pos_type = type_dist_hemi_torch.view(type_dist_hemi_torch.shape[0], 
                                               np.prod(type_dist_hemi_torch.shape[1:4]), 
                                               type_dist_hemi_torch.shape[4])
    
    # Create masks for valid voxels and types
    Vix = torch_nansum(torch_nanmean(animal_pos_type, axis=0), axis=1) > min_voxel_density
    Tix = torch_nansum(torch_nanmean(animal_pos_type, axis=0), axis=0) > min_type_density
    
    # Prepare female-male comparison data
    fm_animal_pos_type_no_nans = animal_pos_type[np.hstack((female_hemi_ix, male_hemi_ix))][:, Vix, :][:, :, Tix]
    fm_animal_pos_type_no_nans[torch.isnan(fm_animal_pos_type_no_nans)] = 0
    fm_animal_pos_type_is_nans = torch.isnan(animal_pos_type[np.hstack((female_hemi_ix, male_hemi_ix))][:, Vix, :][:, :, Tix])
    
    # Prepare ASD-male comparison data
    asd_animal_pos_type_no_nans = animal_pos_type[np.hstack((asd_hemi_ix, male_hemi_ix))][:, Vix, :][:, :, Tix]
    asd_animal_pos_type_no_nans[torch.isnan(asd_animal_pos_type_no_nans)] = 0
    asd_animal_pos_type_is_nans = torch.isnan(animal_pos_type[np.hstack((asd_hemi_ix, male_hemi_ix))][:, Vix, :][:, :, Tix])
    
    return {
        'animal_pos_type': animal_pos_type,
        'Vix': Vix,
        'Tix': Tix,
        'fm_animal_pos_type_no_nans': fm_animal_pos_type_no_nans,
        'fm_animal_pos_type_is_nans': fm_animal_pos_type_is_nans,
        'asd_animal_pos_type_no_nans': asd_animal_pos_type_no_nans,
        'asd_animal_pos_type_is_nans': asd_animal_pos_type_is_nans,
        'female_hemi_ix': female_hemi_ix,
        'male_hemi_ix': male_hemi_ix,
        'asd_hemi_ix': asd_hemi_ix
    }


def run_region_analysis(type_dist_4D, female_hemi_ix, male_hemi_ix, asd_hemi_ix, 
                       save_prefix='region_pval', rnd_iter=1000, use_cuda=True):
    """
    Run complete region-based analysis comparing groups.
    
    Parameters:
    -----------
    type_dist_4D : torch.Tensor
        5D tensor of type distributions
    female_hemi_ix, male_hemi_ix, asd_hemi_ix : np.array
        Hemisphere indices for each group
    save_prefix : str
        Prefix for saving results
    rnd_iter : int
        Number of random iterations for permutation testing
    use_cuda : bool
        Whether to use CUDA for computations
    
    Returns:
    --------
    dict : Dictionary containing analysis results
    """
    # Prepare data
    data = prepare_region_analysis_data(type_dist_4D, female_hemi_ix, male_hemi_ix, asd_hemi_ix)
    
    # Clear CUDA memory before analysis
    if use_cuda:
        clear_cuda()
    
    # Run ASD analysis
    print("Running ASD vs Male analysis...")
    asd_pvals = calc_comp_pvals(
        data['asd_animal_pos_type_no_nans'], 
        data['Vix'], 
        data['Tix'],
        data['asd_hemi_ix'][:len(data['asd_hemi_ix'])//2],  # First half for group 1
        data['male_hemi_ix'][:len(data['male_hemi_ix'])//2],  # First half for group 2
        f'{save_prefix}_asd.npy',
        rnd_iter=rnd_iter,
        use_cuda=use_cuda
    )
    
    # Run Female-Male analysis
    print("Running Female vs Male analysis...")
    fm_pvals = calc_comp_pvals(
        data['fm_animal_pos_type_no_nans'],
        data['Vix'],
        data['Tix'],
        data['female_hemi_ix'][:len(data['female_hemi_ix'])//2],  # First half for group 1
        data['male_hemi_ix'][:len(data['male_hemi_ix'])//2],  # First half for group 2
        f'{save_prefix}_fm.npy',
        rnd_iter=rnd_iter,
        use_cuda=use_cuda
    )
    
    return {
        'asd_pvals': asd_pvals,
        'fm_pvals': fm_pvals,
        'data': data
    }


