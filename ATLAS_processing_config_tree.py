import numpy as np
import logging
import torch
# Basic parameters of imaging
bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'),
            ('RS0175_cy5', 'hybe3', 'FarRed'),
            ('RS0332_cy5', 'hybe6', 'FarRed'),
            ('RSN2336.0_cy5', 'hybe8', 'FarRed'),
            ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
            ('RS0384_cy5', 'hybe10', 'FarRed'),
            ('RS0406_cy5', 'hybe11', 'FarRed'),
            ('RS0451_cy5', 'hybe12', 'FarRed'),
            ('RS0548_cy5', 'hybe14', 'FarRed'),
            ('RS64.0_cy5', 'hybe15', 'FarRed'),
            ('RSN4287.0_cy5', 'hybe16', 'FarRed'),
            ('RSN1252.0_cy5', 'hybe17', 'FarRed'),
            ('RSN9535.0_cy5', 'hybe18', 'FarRed'),
            ('RS740.0_cy5', 'hybe23', 'FarRed'),
            ('RS810.0_cy5', 'hybe24', 'FarRed'),
            ('RS458122_cy5', 'hybe21', 'FarRed'), # (’R25’,’RS458122_cy5’, 'hybe25', 'FarRed',’AACTCCTTATCACCCTACTC’)
            ('RS0083_SS_Cy5', 'hybe5', 'FarRed'), # (’R26’,’RS0083_SS_Cy5’, ’hybe5/bDNA’, ‘FarRed’,’ACACTACCACCATTTCCTAT’)
            ('RS0255_SS_Cy5', 'hybe20', 'FarRed')] # R20

nbits = len(bitmap)
parameters = {}

""" New Microscope Setup"""

parameters['fishdata']='Processing_2024May28'
parameters['hard_overwrite'] = False
parameters['use_scratch'] = False 
parameters['bin'] = 2 # how many pixels to bin to save computational time 2x2 
parameters['stitch_rotate'] = 0# NEW0 # NEW 0 #scope specific rotation but all scopes synced
parameters['stitch_flipud'] = False# NEW False #scope specific flip up down but all scopes synced
parameters['stitch_fliplr'] = True# NEW True #scope specific flip left right but all scopes synced
parameters['segment_gpu'] = False #Broken unless you have cuda tensor set up
parameters['max_registration_shift'] = 200 # binned pixels
parameters['QC_pixel_size'] = 2 # um # size of pixel for saved tifs
parameters['diameter'] = 8 #15 # um #used for segmenting cells in um
parameters['nucstain_channel'] = 'DeepBlue' #Reference Channel
parameters['nucstain_acq'] = 'hybe24' # reference hybe
parameters['total_channel'] = 'FarRed' # Signal channel for segmentation
parameters['total_acq'] = 'all_max' #'hybe25' # Which acq to segment on 
parameters['highpass_function'] = 'rolling_ball'#'gaussian_robust[1,60]'#'spline_min_robust_smooth'#'spline_min_smooth'#'spline_robust_min'#'downsample_quantile_0.1' _smooth
parameters['highpass_sigma'] = 25 #binned pixels
parameters['highpass_smooth_function'] = 'median'
parameters['highpass_smooth'] = 3 #binned pixels
parameters['strip'] = True # use strip image as additional backround 
parameters['model_types'] = ['total'] # total nuclei or cytoplasm segmentation options 
parameters['dapi_thresh'] = 200 #minimum dapi signal for segmentation
parameters['overlap'] = 0.02 # 2% overlap
parameters['fileu_version'] = 2 # Version of fileu to use
parameters['overlap_correction'] = False #Problematic # use overlap to correct for constants in the image
parameters['ncpu'] = 5 # number of threads to use
parameters['metric'] = 'median' # what value to pull from the stiched image for each cell 
parameters['microscope_parameters'] = 'microscope_parameters' # directory name for image parameters FF and constant
parameters['imaging_batch'] = 'hybe' # which FF and constants to average 'acq' 'dataset' 'hybe' #maybe brightness depentant
parameters['process_img_before_FF'] = False # process image before FF correction # dont use
parameters['debug'] = False # more figures printed 
parameters['config_overwrite'] = True # should you overwrite your config file saved in the fishdata path
parameters['overwrite'] = False #False # should you overwrite stitching images
parameters['segment_overwrite'] = False #False # should you overwrite segmentation Masks
parameters['vector_overwrite'] = False #False # should you overwrite pulled vectors
parameters['delete_temp_files'] = True #False # should you delete temporary files to reduce space on server
parameters['overwrite_report']= False # report figures overwrite
parameters['overwrite_louvain'] = False # overwrite louvain unsupervised clustering
parameters['scratch_path_base'] = '/scratchdata2/Processing_tmp' # where to save temporary files
parameters['bitmap'] = bitmap # dont change this its for saving and record keeping