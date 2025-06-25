#!/usr/bin/env python
from ATLAS.Analysis.Classification import *
from ATLAS.Registration.execute import *
from scipy.interpolate import interp1d
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
from ATLAS.Utils import geomu
import argparse
from ATLAS.Utils import basicu
import matplotlib.pyplot as plt
from ATLAS.Processing.Section import *
import os
import sys

def convert_effect_2_delta(perturbation,effect_size,vis=True,radius = 5):
    if perturbation == 'merge':
        return 0
    elif perturbation == 'shift':
        a = np.pad(morphology.disk(radius),math.ceil(effect_size*2), mode='constant', constant_values=0)
        b = np.zeros((a.shape[0],a.shape[1]))
        loc = np.where(a==1)
        b[loc[0]+effect_size,loc[1]] = 1
        c = a+b
        delta = 1-np.mean(c[loc]==2)
    elif perturbation == 'dilation':
        if effect_size%1==0:
            a = np.pad(morphology.disk(radius),math.ceil(effect_size*2), mode='constant', constant_values=0)
            b = morphology.binary_dilation(a, morphology.disk(effect_size))
        else:
            a = np.pad(morphology.disk(radius),math.ceil(effect_size*2), mode='constant', constant_values=0)
            b = morphology.binary_dilation(a, morphology.square(int(effect_size+2)))
        c = a+b
        delta = np.sum(c==1)/np.sum(a==1)
    elif perturbation == 'erosion':
        if effect_size%1==0:
            a = np.pad(morphology.disk(radius),math.ceil(effect_size*2), mode='constant', constant_values=0)
            b = morphology.binary_erosion(a, morphology.disk(effect_size))
        else:
            a = np.pad(morphology.disk(radius),math.ceil(effect_size*2), mode='constant', constant_values=0)
            b = morphology.binary_erosion(a, morphology.square(int(effect_size+2)))
        c = a+b
        delta = 1-(np.sum(c==2)/np.sum(a==1))
    if vis:
        plt.imshow(c)
        plt.title(f"{perturbation} {effect_size} {delta}")
        plt.show()
    return delta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("perturbation", type=str)
    args = parser.parse_args()

if __name__ == '__main__':
    perturbation = args.perturbation

    if not perturbation in ['merge','shift','dilation','erosion']:
        raise ValueError("Perturbation must be one of ['merge','shift','dilation','erosion']")


    """ Without Purterbation """
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR)
    np.random.seed(42)

    # if os.path.exists(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/classified_anndata_{perturbation}_base.h5ad"):
    #     adata_base = anndata.read_h5ad(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/classified_anndata_{perturbation}_base.h5ad")
        
    adata = anndata.read_h5ad("/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/anndata.h5ad")
    animal = 'WTM01'
    adata.obs['animal'] = animal
    adata.obs['dataset'] = 'WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08'
    adata.obs['processing'] = 'Processing_2025Feb04'
    adata
    registration_path = '/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Registration_2024Jul02'
    section_acq_name = 'WellA-Section3'
    XYZC  = Registration_Class(adata.copy(),registration_path,section_acq_name,verbose=False,regularize=True).run()
    adata.obs['ccf_x'] = XYZC['ccf_x']
    adata.obs['ccf_y'] = XYZC['ccf_y']
    adata.obs['ccf_z'] = XYZC['ccf_z']
    adata.obs['old_section_name'] = section_acq_name
    adata.obs['registration_path'] = registration_path
    section_name = f"{animal}_{adata.obs['ccf_x'].mean():.1f}"
    adata.obs['section_name'] = section_name
    adata.obs['Slice'] = section_name

    XY = np.array(adata.obs[["ccf_z","ccf_y"]])

    adata.obs['in_large_comp'] = geomu.in_graph_large_connected_components(XY,Section = None,max_dist = 0.05,large_comp_def = 0.1,plot_comp = False)
    adata = adata[adata.obs['in_large_comp']==True].copy()
    print(f"Keeping {adata.shape[0]} cells after component filtering for {section_acq_name}")
    adata

    n_cells_pre_nuc_filtering = adata.shape[0]
    adata.layers['nuc_mask'] = basicu.filter_cells_nuc(adata)
    adata = adata[np.sum(adata.layers['nuc_mask']==False,axis=1)<2].copy()
    adata = adata[np.clip(np.array(adata.layers['raw']).copy().sum(1),1,None)>100].copy()
    print(f"Keeping {adata.shape[0]} cells after nuc filtering for {section_acq_name}")
    adata

    adata.X = adata.layers['raw'].copy()


    scale = SingleCellAlignmentLeveragingExpectations(adata,visualize=False,verbose=False)
    scale.calculate_spatial_priors()
    scale.load_reference()
    backup_measured1 = scale.measured.copy()
    backup_reference1 = scale.reference.copy()
    scale.model = LogisticRegression(max_iter=1000,random_state=42) 

    scale.supervised_neuron_annotation()

    scale.supervised_harmonization()

    adata_updated = scale.measured.copy()
    # c = adata_updated.obs['subclass_color']
    # plt.figure(figsize=(15,15))
    # plt.scatter(adata_updated.obs['ccf_z'],adata.obs['ccf_y'],c=c,s=2,marker=',', edgecolors='none', linewidths=0)
    # plt.grid('off')
    # plt.axis('off')
    # plt.show()
    adata_base = adata_updated.copy()
    adata_base.write_h5ad(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/classified_anndata_{perturbation}_base.h5ad")


    np.random.seed(42)

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR)
    out_results = {}

    metadata_path = '/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/'
    section = 'WellA-Section3'
    cword_config = 'ATLAS_processing_config_tree_test'
    adata = anndata.read_h5ad("/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/anndata.h5ad")

    if perturbation == 'merge':
        effect_sizes = [0]

        XY = torch.tensor(np.array(adata.obs[['stage_x','stage_y']]))
        cells = torch.arange(XY.shape[0])
        dmat = torch.cdist(XY, XY, p=2)
        dmat[dmat==0] = np.inf
        neighbors_idx = torch.argmin(dmat,axis=0)
        neighbors_neighbors_idx = neighbors_idx[neighbors_idx]
        mutual_mask = cells == neighbors_neighbors_idx
        mutual_cells = cells[mutual_mask]
        mutual_neighbors = neighbors_idx[mutual_mask]

        labels = torch.tensor(np.array(adata.obs['label']))
        mutual_cell_labels = labels[mutual_cells]
        mutual_neighbor_labels = labels[mutual_neighbors]

    elif perturbation == 'shift':
        effect_sizes = [1,2,3,4,5,6,7,8]
    elif perturbation == 'dilation':
        effect_sizes = [0.5,1,1.5,2,2.5,3]
    elif perturbation == 'erosion':
        effect_sizes = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

    for effect_size in effect_sizes:
        for replicate in range(3):
            # if os.path.exists(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/{perturbation}_{effect_size}_rep{replicate}_classified_anndata_base.h5ad")&os.path.exists(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/other/{perturbation}_{effect_size}_rep{replicate}_affected_cells.pt"):
            #     continue
            print(f"Starting {perturbation} {effect_size} {replicate}")
            adata = anndata.read_h5ad("/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/anndata.h5ad")

            """ Generate Mask """
            base_mask = torch.load('/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/mask/mask.pt')
            perturbation_mask = base_mask.clone().detach()
            n_affected_cells = 5000
            cells = torch.tensor(adata.obs['label'].unique())
            if perturbation == 'merge':
                affected_cells_idx = np.random.choice(np.arange(mutual_cell_labels.shape[0]), n_affected_cells, replace=False)
                affected_cells = mutual_cell_labels[affected_cells_idx].numpy()
                affected_neighbor_cells = mutual_neighbor_labels[affected_cells_idx].numpy()
                for i in range(affected_cells.shape[0]):
                    cell = affected_cells[i]
                    neighbor = affected_neighbor_cells[i]
                    cell_size = adata.obs['size'][adata.obs['label']==cell].values[0]
                    neighbor_cell_size = adata.obs['size'][adata.obs['label']==affected_neighbor_cells[i]].values[0]
                    if cell_size<neighbor_cell_size:
                        affected_cells[i] = neighbor
                        affected_neighbor_cells[i] = cell
            else:
                affected_cells = np.random.choice(cells, n_affected_cells, replace=False)

            loc = base_mask>0
            loc_coords = torch.where(loc)
            labels = base_mask[loc]
            window = math.ceil(5+effect_size)

            updated_affected_cells = affected_cells.copy()
            for it,cell in tqdm(enumerate(affected_cells),desc=f"Perturbing {perturbation} {effect_size}"):
                og_cell = cell
                if perturbation == 'merge':
                    neighbor_cell = affected_neighbor_cells[it]
                    cell_idxs = (labels==cell)|(labels==neighbor_cell)
                    perturbation_mask[perturbation_mask==neighbor_cell] = torch.tensor(cell)
                    continue
                else:
                    cell_idxs = labels==cell
                cell_loc_0 = loc_coords[0][cell_idxs]
                cell_loc_1 = loc_coords[1][cell_idxs]
                cell_loc_0_min = cell_loc_0.min()-window
                cell_loc_0_max = cell_loc_0.max()+window
                cell_loc_1_min = cell_loc_1.min()-window
                cell_loc_1_max = cell_loc_1.max()+window
                cell_window = torch.clone(perturbation_mask[cell_loc_0_min:cell_loc_0_max, cell_loc_1_min:cell_loc_1_max].detach())
                updated_cell_window = 0*cell_window.numpy().copy()
                if perturbation == 'erosion':
                    updated_cell_window[cell_window==cell] = 1
                    if effect_size%1==0:
                        updated_cell_window = morphology.binary_erosion(updated_cell_window, morphology.disk(effect_size))
                    else:
                        updated_cell_window = morphology.binary_erosion(updated_cell_window, morphology.square(math.ceil(effect_size+1)))
                elif perturbation == 'dilation':
                    updated_cell_window[cell_window==cell] = 1
                    if effect_size%1==0:
                        updated_cell_window = morphology.binary_dilation(updated_cell_window, morphology.disk(effect_size))
                    else:
                        updated_cell_window = morphology.binary_dilation(updated_cell_window, morphology.square(math.ceil(effect_size+1)))
                elif perturbation == 'shift':
                    axis = np.random.choice([0, 1])
                    direction = np.random.choice([-1, 1])
                    dim1,dim2 = torch.where(cell_window==cell)
                    if axis == 0:
                        dim1 = dim1 + direction*effect_size
                    elif axis == 1:
                        dim2 = dim2 + direction*effect_size
                    updated_cell_window[dim1.numpy(),dim2.numpy()] = 1
                elif perturbation == 'merge':
                    updated_cell_window[cell_window==cell] = 1
                    updated_cell_window[cell_window==neighbor_cell] = 1
                else:
                    raise ValueError("Perturbation not recognized")

                cell_window[cell_window==cell] = 0
                cell_window[torch.tensor(updated_cell_window)==1] = torch.tensor(cell)
                perturbation_mask[cell_loc_0_min:cell_loc_0_max, cell_loc_1_min:cell_loc_1_max] = cell_window
                # break

            torch.save(perturbation_mask, f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/mask/{perturbation}_{effect_size}_rep{replicate}_mask.pt")
            torch.save(torch.tensor(affected_cells),f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/other/{perturbation}_{effect_size}_rep{replicate}_affected_cells.pt")
            if perturbation == 'merge':
                torch.save(torch.tensor(affected_neighbor_cells),f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/other/{perturbation}_{effect_size}_rep{replicate}_neighbor_cells.pt")

            """ Generate Anndata """
            processing = Section_Class(metadata_path,section,cword_config,verbose=False)
            processing.model_type = f"{perturbation}_{effect_size}_rep{replicate}"
            processing.parameters['model_types'] = [processing.model_type,'']

            processing.path = '/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/'
            processing.scratch_path = processing.path

            processing.load_metadata()
            processing.mask = perturbation_mask
            processing.parameters['vector_overwrite'] = True
            processing.pull_vectors()
            adata = processing.data.copy()

            """ Classify """
            animal = 'WTM01'
            adata.obs['animal'] = animal
            adata.obs['dataset'] = 'WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08'
            adata.obs['processing'] = 'Processing_2025Feb04'
            
            registration_path = '/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Registration_2024Jul02'
            section_acq_name = 'WellA-Section3'
            XYZC  = Registration_Class(adata.copy(),registration_path,section_acq_name,verbose=False,regularize=True).run()
            adata.obs['ccf_x'] = XYZC['ccf_x']
            adata.obs['ccf_y'] = XYZC['ccf_y']
            adata.obs['ccf_z'] = XYZC['ccf_z']
            adata.obs['old_section_name'] = section_acq_name
            adata.obs['registration_path'] = registration_path
            section_name = f"{animal}_{adata.obs['ccf_x'].mean():.1f}"
            adata.obs['section_name'] = section_name
            adata.obs['Slice'] = section_name


            XY = np.array(adata.obs[["ccf_z","ccf_y"]])

            adata.obs['in_large_comp'] = geomu.in_graph_large_connected_components(XY,Section = None,max_dist = 0.05,large_comp_def = 0.1,plot_comp = False)
            adata = adata[adata.obs['in_large_comp']==True].copy()
            # print(f"Keeping {adata.shape[0]} cells after component filtering for {section_acq_name}")
            
            n_cells_pre_nuc_filtering = adata.shape[0]
            adata.layers['nuc_mask'] = basicu.filter_cells_nuc(adata)
            adata = adata[np.sum(adata.layers['nuc_mask']==False,axis=1)<2].copy()
            adata = adata[np.clip(np.array(adata.layers['raw']).copy().sum(1),1,None)>100].copy()
            # print(f"Keeping {adata.shape[0]} cells after nuc filtering for {section_acq_name}")
            adata.X = adata.layers['raw'].copy()
            # scale = SingleCellAlignmentLeveragingExpectations(adata,visualize=False,verbose=True)
            scale.measured = adata.copy()
            scale.calculate_spatial_priors()
            # scale.load_reference()
            scale.model = LogisticRegression(max_iter=1000,random_state=42) 
            scale.supervised_neuron_annotation()
            scale.supervised_harmonization()
            scale.measured.write_h5ad(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/{perturbation}_{effect_size}_rep{replicate}_classified_anndata_base.h5ad")


            if perturbation=='merge':
                adata_perturbed = anndata.read_h5ad(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/{perturbation}_{effect_size}_rep{replicate}_classified_anndata_base.h5ad")
                affected_cells = torch.load(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/other/{perturbation}_{effect_size}_rep{replicate}_affected_cells.pt")
                affected_neighbors_cells = torch.load(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/other/{perturbation}_{effect_size}_rep{replicate}_neighbor_cells.pt")
                accuracy = []
                for idx in trange(affected_cells.shape[0]):
                    cell = affected_cells[idx].numpy()
                    neighbor = affected_neighbors_cells[idx].numpy()
                    try:
                        correct_call = adata_base[adata_base.obs['label']==cell].obs['subclass'].values[0]
                        correct_neghbor = adata_base[adata_base.obs['label']==neighbor].obs['subclass'].values[0]
                        perturbed = adata_perturbed[adata_perturbed.obs['label']==cell].obs['subclass'].values[0]
                    except:
                        continue
                    accuracy.append((correct_call==perturbed)|(correct_neghbor==perturbed))
                # print(len(accuracy))
                accuracy = np.mean(accuracy)
                out_results[f"{perturbation}_{effect_size}_rep{replicate}"] = accuracy
                print(f"----------{perturbation}_{effect_size}_rep{replicate} : {accuracy}----------")
            else:
                adata_perturbed = anndata.read_h5ad(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/anndata/{perturbation}_{effect_size}_rep{replicate}_classified_anndata_base.h5ad")
                affected_cells = torch.load(f"/scratchdata1/Images2024/Zach/MouseBrainAtlas/WTM01_3.2.A_2.2.B_1.2.C_6.2.D_5.2.E_4.2.F_2024Apr08/Processing_2025Feb04/WellA-Section3/other/{perturbation}_{effect_size}_rep{replicate}_affected_cells.pt")
                shared_cells = np.intersect1d(adata_base[adata_base.obs['label'].isin(affected_cells.numpy())].obs.index,adata_perturbed.obs.index)
                adata1 = adata_base[shared_cells,:].copy()
                adata2 = adata_perturbed[shared_cells,:].copy()
                accuracy = np.mean(np.array(adata1.obs['subclass'])==np.array(adata2[adata1.obs.index].obs['subclass']))
                out_results[f"{perturbation}_{effect_size}_rep{replicate}"] = accuracy
                print(f"----------{perturbation}_{effect_size}_rep{replicate} : {accuracy}----------")

