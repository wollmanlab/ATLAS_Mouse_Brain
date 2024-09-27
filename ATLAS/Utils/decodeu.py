import os
import shutil
from ATLAS.Analysis.Classification import *
from ATLAS.Visualization.Viz import *
from ATLAS.Registration.Registration import *
from ATLAS.Utils import fileu, basicu, ccfu
import gc
import traceback
import imageio
import warnings 
from sklearn.exceptions import ConvergenceWarning


def analyze_mouse_brain_data(animal,
                            project_path='/scratchdata1/Images2024/Zach/MouseBrainAtlas',
                            analysis_path='/scratchdata1/MouseBrainAtlases_V1',
                            verbose=False,repair=False,register_to_ccf=True,ccf_x_min=-np.inf,ccf_x_max=np.inf):
    """ 
    Analyze mouse brain data and perform various analysis steps.
    
    Parameters:
    - animal: str, the name of the animal to analyze
    - project_path: str, the path to the project directory (default: '/scratchdata1/Images2024/Zach/MouseBrainAtlas')
    - analysis_path: str, the path to the analysis directory (default: '/scratchdata1/MouseBrainAtlases_V1')
    - verbose: bool, whether to print verbose output (default: False)
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    try:

        # Create analysis directory for the animal
        input_df = fileu.create_input_df(project_path, animal)
        basepath = os.path.join(analysis_path, animal)
        figure_path = os.path.join(basepath,'Figures')
        tax_basepath = os.path.join(analysis_path, "Taxonomies")
        adata_path = os.path.join(basepath,'Layer')
        if not os.path.exists(adata_path):
            os.makedirs(adata_path,mode=0o777)
        adata_file_name = os.path.join(adata_path,'cell_layer.h5ad')
        if os.path.exists(adata_file_name):
            adata = anndata.read_h5ad(adata_file_name)
        else:
            adatas = []
            shared_bits = ''
            used_names = []
            fileu.update_user(f"Attempting To Load {input_df.shape[0]} Sections")
            for index,row in tqdm(input_df.iterrows(),desc='Loading Sections',total=input_df.shape[0]):
                animal = row['animal']
                section_acq_name = row['section_acq_name']
                dataset = row['dataset']
                registration_path = row['registration_path']
                processing = row['processing']
                dataset_path = row['dataset_path']
                if not os.path.exists(os.path.join(dataset_path,dataset,processing,section_acq_name)):
                    fileu.update_user(f" Processing path Not Found {section_acq_name} {os.path.join(dataset_path,dataset,processing,section_acq_name)}")
                    continue
                if not os.path.exists(os.path.join(registration_path,section_acq_name)):
                    fileu.update_user(f" Registration path Not Found {section_acq_name} {os.path.join(registration_path,section_acq_name)}")
                    continue
                try:
                    adata = fileu.load(os.path.join(dataset_path,dataset,processing,section_acq_name),file_type='anndata')
                except:
                    fileu.update_user(f"Unable to Load Data {section_acq_name}")
                    continue
                if adata.shape[0]<100:
                    continue
                adata.obs['animal'] = row['animal']
                adata.obs['dataset'] = row['dataset']
                adata.obs['processing'] = row['processing']
                if register_to_ccf: 
                    try:
                        XYZC  = Registration_Class(adata.copy(),registration_path,section_acq_name,verbose=False,regularize=True).run()
                        adata.obs['ccf_x'] = XYZC['ccf_x']
                        adata.obs['ccf_y'] = XYZC['ccf_y']
                        adata.obs['ccf_z'] = XYZC['ccf_z']
                        """ Rename Section """
                        if adata.obs['ccf_x'].mean() < ccf_x_min:
                            fileu.update_user(f"Outside ccf window {section_acq_name} {adata.obs['ccf_x'].mean()}")
                            continue
                        elif adata.obs['ccf_x'].mean() > ccf_x_max:
                            fileu.update_user(f"Outside ccf window {section_acq_name} {adata.obs['ccf_x'].mean()}")
                            continue
                        section_name = f"{animal}_{adata.obs['ccf_x'].mean():.1f}"
                        adata.obs['old_section_name'] = section_acq_name
                        adata.obs['registration_path'] = row['registration_path']
                    except:
                        fileu.update_user(f"Unable to Register Data {section_acq_name}")
                        continue
                else: 
                    section_name = section_acq_name
                i = 1
                base_section_name =section_name
                while section_name in used_names:
                    section_name = f"{base_section_name}_{i}"
                    i += 1
                adata.obs['section_name'] = section_name
                adata.obs[adata_mapping['Section']] = section_name
                if isinstance(shared_bits,str):
                    shared_bits = list(adata.var.index)
                else:
                    shared_bits = [i for i in shared_bits if i in list(adata.var.index)]
                if register_to_ccf: 
                    XY = np.array(adata.obs[["ccf_z","ccf_y"]])
                else: 
                    XY = np.array(adata.obs[["stage_x","stage_y"]])
                fileu.update_user(f"Found {adata.shape[0]} cells for {section_acq_name}")
                bad_bits = ['RS0109_cy5','RSN9927.0_cy5','RS0468_cy5','RS643.0_cy5','RS156.0_cy5','RS0237_cy5']
                adata = adata[:,np.isin(adata.var.index,bad_bits,invert=True)].copy()

                adata.obs['in_large_comp'] = geomu.in_graph_large_connected_components(XY,Section = None,max_dist = 0.05,large_comp_def = 0.1,plot_comp = False)
                adata = adata[adata.obs['in_large_comp']==True].copy()
                fileu.update_user(f"Keeping {adata.shape[0]} cells after component filtering for {section_acq_name}")

                if adata.shape[0]<100:
                    fileu.update_user(f" Not Enough Cells Found {adata.shape[0]} cells")
                    fileu.update_user(f"Look Into Section {section_acq_name}")
                    continue

                n_cells_pre_nuc_filtering = adata.shape[0]
                adata.layers['nuc_mask'] = basicu.filter_cells_nuc(adata)
                adata = adata[np.sum(adata.layers['nuc_mask']==False,axis=1)<2].copy()
                adata = adata[np.clip(np.array(adata.layers['raw']).copy().sum(1),1,None)>100].copy()

                if adata.shape[0]<100:
                    fileu.update_user(f" Not Enough Cells Found {adata.shape[0]} cells")
                    fileu.update_user(f"Look Into Section {section_acq_name}")
                    continue

                fileu.update_user(f"Keeping {adata.shape[0]} cells after nuc filtering for {section_acq_name}")
                n_cells_post_nuc_filtering = adata.shape[0]
                if n_cells_post_nuc_filtering < 0.6 * n_cells_pre_nuc_filtering:
                    fileu.update_user(f"Likely Gel issues: More than 40% cells Removed {adata.shape[0]} cells")
                    fileu.update_user(f"Look Into Section {section_acq_name}")
                    # continue
                if adata.shape[0]<60000:
                    fileu.update_user(f" Not Enough Cells Found {adata.shape[0]} cells")
                    fileu.update_user(f"Look Into Section {section_acq_name}")
                    # continue

                adata.X = adata.layers['raw'].copy()

                if adata.shape[0]<100:
                    fileu.update_user(f" Not Enough Cells Found {adata.shape[0]} cells")
                    fileu.update_user(f"Look Into Section {section_acq_name}")
                    continue
                
                adata.X = basicu.normalize_fishdata_robust_regression(adata.X.copy())

                
                adata.X = basicu.image_coordinate_correction(adata.X.copy(),np.array(adata.obs[["image_x","image_y"]]))

                adata.layers['normalized'] = adata.X.copy()

                adata.layers['classification_space'] = basicu.robust_zscore(adata.X.copy())
                animal_adata = adata.copy()
                animal_adata.write(adata_file_name)

        # order sections
        def get_ccf_x(section):
            return float(re.split('[_\.]', section)[1])
        unqS =  sorted(np.unique(animal_adata.obs['Slice'].unique()), key=get_ccf_x)
        level = 'subclass'
        if not level in animal_adata.obs.columns:
            seq_adata = anndata.read_h5ad(pathu.get_path('allen_wmb_tree'))
            seq_adata.index = [i.split('raise')[0] for i in seq_adata.obs.index]
            pallette = dict(zip(seq_adata.obs[level],seq_adata.obs[f"{level}_color"]))
            
            fileu.update_user(f"Harmonizing to Reference and Classifying", verbose=True)
            out = []
            for idx,section in enumerate(unqS):
                fileu.update_user(f" {section} {idx}  out of {len(unqS)}")
                m = animal_adata.obs['Slice'] == section
                if np.sum(m) == 0:
                    continue
                idx = np.where(m)[0]
                adata = animal_adata[idx]
                scale = SingleCellAlignmentLeveragingExpectations(adata,visualize=False,verbose=False)
                scale.complete_reference = seq_adata
                adata_updated = scale.run()
                out.append(adata_updated)
                print(adata_updated)
            adata = anndata.concat(out)
            adata = adata[animal_adata.obs.index].copy()
            for layer in ['harmonized','imputed','zscored']:
                animal_adata.layers[layer] = adata.layers[layer]

            for label in ['subclass','leiden','neuron']:
                animal_adata.obs[label] = adata.obs[label].astype(str)
                animal_adata.obs[label+'_color'] = adata.obs[label+'_color'].astype(str)

            neighbor_columns = [i for i in adata.obs.columns if 'neighbor' in i]
            for col in neighbor_columns:
                animal_adata.obs[col] = adata.obs[col]
            
            print(animal_adata)
            animal_adata.write(adata_file_name)
            gc.collect()

        
        fileu.update_user("Completed", verbose=verbose)
    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        print(animal)
        print('Failed')
        fileu.update_user("Failed", verbose=verbose)
        fileu.update_user(str(e), verbose=verbose)
        fileu.update_user(str(error_message), verbose=verbose)

