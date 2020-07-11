import os

# ===================== Visualization Settings =============================
port = 8095
env = 'main'
same_env = True
# ===================== Visualization Settings =============================

# ======================== Main Setings ====================================
log_type = 'traditional'
train = 'face_cls'
save_results = True
result_path = '/research/prip-gongsixu/codes/biasface/results/models/gac/balance_gac_test'
extract_feat = False
just_test = False
feat_savepath = '/research/prip-gongsixu/codes/biasface/results/features/feat_lfw_base.npz'
resume = None
# resume = '/research/prip-gongsixu/codes/biasface/results/models/balance_attrace_weight1.0/Save'
# resume = '/research/prip-gongsixu/results/models/teacher/facenet_vgg2.pth'
# resume = '/scratch/gongsixue/face_resolution/models/model_ir_se50.pth'
# resume = '/research/prip-gongsixu/codes/biasface/results/models/face_arc_l50_h5py_4/Save'
# ======================== Main Setings ====================================

# ======================= Data Setings =====================================
dataset_root_test = None
# dataset_root_test = '/research/prip-gongsixu/datasets/LFW/lfw_aligned_retina_112'
# dataset_root_test = '/user/pripshare/Databases/FaceDatabasesPublic'
dataset_root_train = '/research/prip-gongsixu/datasets/RFW'

# preprocessing
preprocess_train = {"Resize": True, 
    "CenterCrop": True,
    # "RandomCrop": "True",
    "RandomHorizontalFlip": True, 
    # "RandomVerticalFlip": True,  
    # "RandomRotation": 10,
    "Normalize": ((0.5,0.5,0.5), (0.5,0.5,0.5)), 
    "ToTensor": True}

preprocess_test = {"Resize": True, 
    "CenterCrop": True, 
    # "RandomCrop": True, 
    # "RandomHorizontalFlip": True, 
    # "RandomVerticalFlip": True, 
    # "RandomRotation": 10, 
    "Normalize": ((0.5,0.5,0.5), (0.5,0.5,0.5)), 
    "ToTensor": True}

loader_input = 'loader_image'
loader_label = 'loader_numpy'

# dataset_train = 'FileListLoader'
dataset_train = 'CSVListLoader'
# dataset_train = 'H5pyLoader'
# dataset_train = 'ClassSamplesDataLoader'
input_filename_train = '/research/prip-gongsixu/datasets/RFW/attr_rfw_balance_aligned_112.txt'
# input_filename_train = ['/scratch/gongsixue/msceleb_AlignedAsArcface_images.hdf5',\
#     '/research/prip-gongsixu/codes/biasface/datasets/list_faces_emore.txt']
label_filename_train = None
dataset_options_train = {'ifile':input_filename_train, 'ind_attr':[2], 'root':dataset_root_train,
                 'transform':preprocess_train, 'loader':loader_input}

ndemog = 4
# num_images = 75
# dataset_options_train = {'root':dataset_root_train, 'ifile':input_filename_train, 'num_images':num_images, \
#     'ind_attr':[2], 'transform':preprocess_train, 'loader':loader_input}

dataset_test = 'CSVListLoader'
# dataset_test = 'FileListLoader'
input_filename_test = '/research/prip-gongsixu/datasets/RFW/attr_rfw_test_Black_aligned_112.txt'
# input_filename_test = '/research/prip-gongsixu/datasets/LFW/list_lfw_aligned_retina_112.txt'
label_filename_test = None
dataset_options_test = {'ifile':input_filename_test, 'ind_attr':[2], 'root':dataset_root_test,
                 'transform':preprocess_test, 'loader':loader_input}

save_dir = os.path.join(result_path,'Save')
logs_dir = os.path.join(result_path,'Logs')
# ======================= Data Setings =====================================

# ======================= Network Model Setings ============================
# cpu/gpu settings
cuda = True
ngpu = 2
nthreads = 1

nclasses = 28000 # balance
# nclasses = 38737 # unbalance
# nclasses = 75460 # ms1m wo rfw
# nclasses = 85742 # cleaned ms1m

# model_type = 'resnet18'
# model_options = {"nchannels":3,"nfeatures":512}
# model_type = 'incep_resnetV1'
# model_options = {"classnum": 10575, "features":False}
# model_type = 'sphereface20'
# model_options = {"nchannels":3, "nfilters":64, \
#     "ndim":512, "nclasses":nclasses, "dropout_prob":0.4, "features":False}

# model_type = 'resnet_face50'
# model_options = {"nclasses": nclasses}
# model_type = 'attdemog_face50'
model_type = 'gac_face50'
use_spatial_att = False
att_weight = 1.0
fuse_epoch = 1
gac_threshold = 0.1
model_options = {"use_spatial_att":use_spatial_att, "nclasses": nclasses, "ndemog":ndemog,\
    "hard_att_channel":False, "hard_att_spatial":False, \
    "lowresol_set":{'rate':1.0, 'mode':'nearest'},\
    "fuse_epoch":fuse_epoch}
# mode: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'

# model_type = 'Backbone'
# model_options = {"num_layers": 50, "drop_ratio": 0.6, "mode": 'ir_se'}

# loss_type = 'Classification'
# loss_options = {"if_cuda":cuda}
# loss_type = 'BinaryClassify'
# loss_options = {"weight_file":'/research/prip-gongsixu/codes/biasface/datasets/weights_binary_classifier.npy',\
#     "if_cuda":cuda}
# loss_type = 'Regression'
# loss_options = {}
# loss_type = 'AM_Softmax_marginatt'
# loss_options = {"nfeatures":512, "nclasses":nclasses, "s":64.0, "lambda_regular":50, "ndemog":ndemog}
loss_type = "AM_Softmax_arcface"
loss_options = {"nfeatures":512, "nclasses":nclasses, "s":64.0, "m":0.35}

# input data size
input_high = 112
input_wide = 112
resolution_high = 112
resolution_wide = 112
# ======================= Network Model Setings ============================

# ======================= Training Settings ================================
# initialization
manual_seed = 0
nepochs = 300
epoch_number = 0

# batch
batch_size = 200
test_batch_size = 100

# optimization
# optim_method = 'Adam'
# optim_options = {"betas": (0.9, 0.999)}
optim_method = "SGD"
optim_options = {"momentum": 0.9, "weight_decay": 5e-4}

# learning rate
learning_rate = 1e-1
# scheduler_method = 'CosineAnnealingLR'
scheduler_method = 'Customer'
scheduler_options = {"T_max": nepochs, "eta_min": 1e-6}
# lr_schedule = [8, 13, 15]
lr_schedule = [6, 13, 15]
# lr_schedule = [33368,54230,62580]
# ======================= Training Settings ================================

# ======================= Evaluation Settings ==============================
# label_filename = os.path.join('/research/prip-gongsixu/results/feats/evaluation', 'list_lfwblufr.txt')
label_filename = input_filename_test

# protocol and metric
protocol = 'LFW'
metric = 'cosine'

# files related to protocols
# IJB
eval_dir = '/research/prip-gongsixu/results/evaluation/ijbb/sphere/cs3'
# eval_dir = '/research/prip-gongsixu/results/evaluation/ijba'
imppair_filename = os.path.join(eval_dir, 'imp_pairs.csv')
genpair_filename = os.path.join(eval_dir, 'gen_pairs.csv')
pair_index_filename={'imposter':imppair_filename,'genuine':genpair_filename}
# pair_index_filename = eval_dir
template_filename = os.path.join(eval_dir, 'temp_dict.pkl')

# LFW
pairs_filename = '/research/prip-gongsixu/results/evaluation/lfw/lfw_pairs_rm1line.txt'
nfolds=10

# RFW
pairs_filename = '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/African/African_pairs.txt'
# pairs_filename = {'African': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/African/African_pairs.txt',\
#     'Asian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Asian/Asian_pairs.txt',\
#     'Caucasian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Caucasian/Caucasian_pairs.txt',\
#     'Indian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Indian/Indian_pairs.txt'}

# features saved as npm
nimgs=None
ndim=None

evaluation_type = 'FaceVerification'
evaluation_options = {'label_filename': label_filename,\
    'protocol': protocol, 'metric': metric,\
    'nthreads': nthreads, 'multiprocess':True,\
    'pair_index_filename': pair_index_filename,'template_filename': template_filename,\
    'pairs_filename': pairs_filename, 'nfolds': nfolds,\
    'nimgs': nimgs, 'ndim': ndim}

# evaluation_type = 'Top1Classification'
# evaluation_type = 'BiOrdinalClassify'
# evaluation_type = 'Agergs_classification'
# evaluation_options = {}
# ======================= Evaluation Settings ==============================
