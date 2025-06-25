# ATLAS_Mouse_Brain

ATLAS (Atlas scale Trasnscriptome Localization using Aggregate Signatures) for Mouse Brain is a code companion to the preprint (https://www.biorxiv.org/content/10.1101/2024.10.08.617260v1) for processing and analyzing ATLAS spatial transcriptomics data from mouse brain tissue.

## Installation

### Prerequisites

- Conda (Anaconda or Miniconda)
- Python 3.9.0
- CUDA-compatible GPU (optional, for PyTorch GPU acceleration)

### Setup Instructions

1. **Create a new conda environment:**
   ```bash
   conda create -n "ATLAS_3.9" python=3.9.0
   conda activate ATLAS_3.9
   ```

2. **Install required packages:**
   ```bash
   # Core scientific computing packages
   pip install scanpy==1.9.3
   pip install numpy==1.23.3
   pip install scikit-image==0.19.3
   
   # Image processing and analysis
   pip install cellpose==3.0.4
   pip install Shapely==1.8.2
   pip install pynrrd==0.4.2
   pip install rasterio==1.2.10
   
   # Graph analysis and clustering
   pip install igraph==0.10.6
   pip install pynndescent==0.5.5
   pip install louvain==0.8.1
   
   # Visualization and dimensionality reduction
   pip install datashader==0.16.0
   pip install umap-learn==0.5.3
   
   # Utilities
   pip install dill==0.3.4
   pip install ashlar==1.17.0
   ```

3. **Install PyTorch (GPU version recommended):**
   ```bash
   # Remove any existing torch installation
   pip uninstall torch
   
   # Install PyTorch with CUDA support (for GPU acceleration)
   conda install pytorch=1.13.0 torchvision=0.14.0 torchaudio=0.13.0 cudatoolkit=11.3 -c pytorch
   
   # For CPU-only installation, use:
   # conda install pytorch=1.13.0 torchvision=0.14.0 torchaudio=0.13.0 cpuonly -c pytorch
   ```

4. **Install the ATLAS package in development mode:**
   ```bash
   # Navigate to the project directory
   cd /path/to/ATLAS_Mouse_Brain
   
   # Install in development mode
   pip install -e .
   ```

### Alternative: Automated Setup

You can run the provided setup script:
```bash
# Make the script executable (if needed)
chmod +x env_setup

# Run the setup script
./env_setup
```

## Running on Data

### Data Requirements

ATLAS expects raw imaging data organized in the following structure:
```
/path/to/raw/data/
├── microscope_parameters/          # Flat field and constant corrections
├── hybe1/                         # Hybridization round 1
│   ├── Position_001/              # Position 1
│   │   ├── hybe1_FarRed_001.tif   # Channel images
│   │   ├── hybe1_DeepBlue_001.tif
│   │   └── ...
│   ├── Position_002/
│   └── ...
├── hybe2/                         # Hybridization round 2
└── ...
```

### Configuration

1. **Modify the configuration file:**
   ```bash
   # Edit ATLAS_processing_config_tree.py to match your data
   nano ATLAS_processing_config_tree.py
   ```

   Key parameters to adjust:
   - `metadata_path`: Path to your raw data directory
   - `bitmap`: Define your probe-to-hybridization mapping
   - `nucstain_acq`: Reference hybridization for registration
   - `total_channel`: Channel used for cell segmentation
   - `ncpu`: Number of CPU cores to use

2. **Example configuration for custom data:**
   ```python
   # Define your probe mapping
   bitmap = [
       ('Probe1_cy5', 'hybe1', 'FarRed'),
       ('Probe2_cy5', 'hybe2', 'FarRed'),
       # ... add all your probes
   ]
   
   # Set data paths
   parameters['metadata_path'] = '/path/to/your/raw/data'
   parameters['fishdata'] = 'Processing_YourDataset'
   ```

### Processing Pipeline

The ATLAS pipeline consists of three main steps:

#### 1. Data Processing
```bash
# Activate environment
conda activate ATLAS_3.9

# Run processing on all sections
python ATLAS/Processing/execute.py /path/to/raw/data -c ATLAS_processing_config_tree

# Process specific section
python ATLAS/Processing/execute.py /path/to/raw/data -c ATLAS_processing_config_tree -s Section1

# Process specific well (if multiple wells)
python ATLAS/Processing/execute.py /path/to/raw/data -c ATLAS_processing_config_tree -w A
```

#### 2. Image Registration
```bash
# Register processed sections
python ATLAS/Registration/execute.py /path/to/raw/data -c ATLAS_processing_config_tree

# Register specific section
python ATLAS/Registration/execute.py /path/to/raw/data -c ATLAS_processing_config_tree -s Section1
```

#### 3. Data Analysis
```bash
# Analyze complete dataset
python ATLAS/Decoding/execute.py animal_name -p /path/to/project -a /path/to/analysis/output

# Example
python ATLAS/Decoding/execute.py Mouse001 -p /scratchdata1/Images2024/Zach/MouseBrainAtlas -a /scratchdata1/MouseBrainAtlases_V3
```

### Output Structure

After processing, you'll find:
```
/path/to/raw/data/
├── Processing_YYYYMMDD/           # Processed data
│   ├── Section1/
│   │   ├── anndata.h5ad          # Final processed data
│   │   ├── processing_log.txt     # Processing log
│   │   └── ...
│   └── ...
├── Registration_YYYYMMDD/         # Registration results
└── ...
```

### Key Parameters

- **Cell Segmentation**: Uses Cellpose with nuclear stain (DeepBlue channel)
- **Image Stitching**: Automatic tile stitching with registration
- **Signal Extraction**: Median intensity per cell per probe
- **Quality Control**: Automatic filtering of low-quality cells

### Troubleshooting

1. **Memory Issues**: Reduce `ncpu` parameter or use `parameters['use_scratch'] = True`
2. **Registration Failures**: Check `max_registration_shift` parameter
3. **Missing Data**: Verify all hybridization rounds are present
4. **GPU Issues**: Set `parameters['segment_gpu'] = False` for CPU-only processing

### Performance and Runtime

**Expected Processing Times:**
- **Desktop Computer (8-16 cores, 32-64GB RAM)**: <24 hours per brain section (<1000 images)


**Parallelization Options:**
- **Multi-core Processing**: Adjust `ncpu` parameter in configuration (default: 5)
- **Multi-section Processing**: Process multiple sections simultaneously by running separate instances
- **Cluster Deployment**: Each section can be processed independently on different compute nodes

**Memory Requirements:**
- **Minimum**: 16GB RAM for single section processing
- **Recommended**: 32GB+ RAM for optimal performance
- **Large Datasets**: Use `parameters['use_scratch'] = True` for temporary file management

**Optimization Tips:**
- Increase `ncpu` to match your available CPU cores
- Use SSD storage for faster I/O operations
- Enable GPU acceleration for cell segmentation if available
- Process sections in parallel on separate machines for large datasets

## Usage

After installation, activate the conda environment:
```bash
conda activate ATLAS_3.9
```

## Project Structure

- `ATLAS/` - Main package directory
  - `Design/` - Probe design and encoding tools
  - `Processing/` - Data processing modules
  - `Registration/` - Image registration tools
  - `Decoding/` - Decoding algorithms
  - `Utils/` - Utility functions
- `ATLAS_processing_config_tree.py` - Configuration file for processing pipeline

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

