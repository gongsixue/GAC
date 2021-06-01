import os
import torchvision.transforms as transforms

import datasets

# ======================== Main Setings ====================================
# train = 'base'
train = 'gac'
save_results =  True
result_path = '/research/prip-gongsixu/codes/biasface/results/models/gac/test_gac_resnet50'
extract_feat = False
just_test = False
feat_savepath = '/research/prip-gongsixu/codes/biasface/results/features/feat_lfw_base.npz'
resume = None

log_type = 'TensorBoard'
tblog_dir = os.path.join(result_path, 'tblog')

if save_results:
    # save_adaptive_distance = True
    save_adaptive_distance = False
else:
    save_adaptive_distance = False
# ======================== Main Setings ====================================

# ======================= Data Setings =====================================
dataset_root_train = '/research/prip-gongsixu/datasets/RFW'
dataset_root_test = None

# input data size
image_height = 112
image_width = 112
image_size = (image_height, image_width)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
# preprocessing
preprocess_train = transforms.Compose([ \
        transforms.CenterCrop(image_size), \
        transforms.Resize(image_size), \
        # transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)), \
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2), \
        # transforms.RandomApply([datasets.loaders.GaussianBlur([.1, 2.])], p=0.5), \
        transforms.RandomHorizontalFlip(), \
        # transforms.RandomVerticalFlip(), \
        # transforms.RandomRotation(10), \
        transforms.ToTensor(), \
        normalize \
    ])

preprocess_test = transforms.Compose([ \
        transforms.CenterCrop(image_size), \
        transforms.Resize(image_size), \
        # transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)), \
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2), \
        # transforms.RandomApply([datasets.loaders.GaussianBlur([.1, 2.])], p=0.5), \
        # transforms.RandomHorizontalFlip(), \
        # transforms.RandomVerticalFlip(), \
        # transforms.RandomRotation(10), \
        transforms.ToTensor(), \
        normalize \
    ])

loader_input = 'loader_image'
loader_label = 'loader_numpy'

# dataset_train = 'FileListLoader'
dataset_train = 'CSVListLoader'
input_filename_train = '/research/prip-gongsixu/datasets/RFW/attr_rfw_balance_aligned_112.txt'
dataset_options_train = {'ifile':input_filename_train, 'ind_attr':[2], 'root':dataset_root_train,
                'transform':preprocess_train, 'loader':loader_input}

# dataset_train = 'H5pyLoader'
# input_filename_train = ['/scratch/gongsixue/msceleb_AlignedAsArcface_images.hdf5',\
#     '/research/prip-gongsixu/codes/biasface/datasets/list_faces_emore.txt']

# dataset_train = 'ClassSamplesDataLoader'
# num_images = 75
# dataset_options_train = {'root':dataset_root_train, 'ifile':input_filename_train, 'num_images':num_images, \
#     'ind_attr':[2], 'transform':preprocess_train, 'loader':loader_input}

dataset_test = 'CSVListLoader'
# dataset_test = 'FileListLoader'
input_filename_test = '/research/prip-gongsixu/datasets/RFW/attr_rfw_test_Black_aligned_112.txt'
# input_filename_test = '/research/prip-gongsixu/datasets/LFW/list_lfw_aligned_retina_112.txt'
dataset_options_test = {'ifile':input_filename_test, 'ind_attr':[2], 'root':dataset_root_test,
                 'transform':preprocess_test, 'loader':loader_input}

save_dir = os.path.join(result_path,'Save')
logs_dir = os.path.join(result_path,'Logs')
# ======================= Data Setings =====================================

# ======================= Network Model Setings ============================
# cpu/gpu settings
cuda = True
ngpu = 4
nthreads = 1

ndemog = 4

nclasses = 28000 # balance
# nclasses = 38737 # unbalance
# nclasses = 75460 # ms1m wo rfw
# nclasses = 85742 # cleaned ms1m

# model_type = 'resnet_face50'
# model_options = {"nclasses": nclasses}

# model_type = 'IR_SE_50'
# model_options = {"input_size": image_size}

model_type = 'gac_resnet50'
# model_type = 'gac_IR_SE_50'
use_spatial_att = False
fuse_epoch = 0

model_options = {"use_spatial_att":use_spatial_att, "nclasses": nclasses, "ndemog":ndemog,\
    "hard_att_channel":False, "hard_att_spatial":False, \
    "lowresol_set":{'rate':1.0, 'mode':'nearest'},\
    "fuse_epoch":fuse_epoch}
#### mode: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' ###

# hyper-parameters for gac
att_weight = 0.5
deb_weight = 0.0
# gac_threshold = 0.0
gac_threshold = 0.1
# gac_threshold = 0.2

loss_type = {"idLoss": "ArcFace"}
loss_options = {"idLoss": {"in_features":512, "out_features":nclasses, "s":64.0, "m":0.50}}


# ======================= Network Model Setings ============================

# ======================= Training Settings ================================
# initialization
manual_seed = 0
nepochs = 256
epoch_number = 0

# batch
batch_size = 300
test_batch_size = 100

# optimization
# optim_method = 'Adam'
# optim_options = {"betas": (0.9, 0.999)}
optim_method = "SGD"
optim_options = {"momentum": 0.9, "weight_decay": 5e-4} # "momentum": 0.9, "weight_decay": 5e-4

# learning rate
learning_rate = 1e-1
# scheduler_method = 'CosineAnnealingLR'
scheduler_method = 'Customer'
scheduler_options = {"T_max": nepochs, "eta_min": 1e-6}
# lr_schedule = [8, 13, 15]
lr_schedule = [5, 17, 19]
# ======================= Training Settings ================================

# ======================= Evaluation Settings ==============================
label_filename = input_filename_test

# protocol and metric
# protocol = 'LFW'
protocol = 'RFW_one_race'
# protocol = 'RFW'
metric = 'cosine'

# LFW
pairs_filename = '/research/prip-gongsixu/results/evaluation/lfw/lfw_pairs.txt'
nfolds=10

# RFW
pairs_filename = {'African': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/African/African_pairs.txt',\
    'Asian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Asian/Asian_pairs.txt',\
    'Caucasian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Caucasian/Caucasian_pairs.txt',\
    'Indian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Indian/Indian_pairs.txt',\
    }

evaluation_type = 'FaceVerification'
evaluation_options = {'label_filename': label_filename,\
    'protocol': protocol, 'metric': metric,\
    'pairs_filename': pairs_filename, 'nfolds': nfolds,\
    }

# evaluation_type = 'Top1Classification'
# evaluation_type = 'BiOrdinalClassify'
# evaluation_type = 'Agergs_classification'
# evaluation_options = {}
# ======================= Evaluation Settings ==============================
