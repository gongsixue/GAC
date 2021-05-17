import os
import numpy as np
import scipy.io
import math
import pickle
import time
import multiprocessing
from functools import partial
from sklearn.model_selection import KFold
from .eval_utils import *
import pdb

class FaceVerification:
    def __init__(self, label_filename,
        protocol='BLUFR', metric='cosine',
        pairs_filename=None,nfolds=10,
        ):

        self.protocol = protocol
        self.metric = metric
        self.nfolds = nfolds

        if label_filename.endswith('npy'):
            self.label = np.load(label_filename)
        elif label_filename.endswith('txt') or label_filename.endswith('csv'):
            self.label = get_labels_from_txt(label_filename)
        else:
            raise(RuntimeError('Format doest not support!'))

        if self.protocol == 'LFW':
            assert(pairs_filename is not None)
            pairfiles = read_pairfile(pairs_filename, self.protocol)
            index_dict = get_index_dict(label_filename)
            self.issame_label,self.pair_indices = get_pair_and_label(pairfiles, index_dict)

        if self.protocol == 'RFW_one_race':
            assert(pairs_filename is not None)
            key = list(pairs_filename)[0]
            pairfiles = read_pairfile(pairs_filename[key], self.protocol)
            index_dict = get_index_dict_rfw(label_filename)
            self.issame_label,self.pair_indices = get_pair_and_label_rfw(pairfiles, index_dict, key)

        if self.protocol == 'RFW':
            assert(pairs_filename is not None)
            index_dict = get_index_dict_rfw(label_filename)
            self.issame_label = {}
            self.pair_indices = {}
            keys = list(pairs_filename)
            race_gender_dict = {'af': 'Asian', 'am': 'Asian', 'bf': 'African', 'bm': 'African',\
                'cf': 'Caucasian', 'cm': 'Caucasian', 'if': 'Indian', 'im': 'Indian'}
            for key in keys:
                pair_file = pairs_filename[key]
                pairfiles = read_pairfile(pair_file, self.protocol)
                issame, pair_indices = get_pair_and_label_rfw(pairfiles, index_dict, key)
                self.issame_label[key] = issame
                self.pair_indices[key] = pair_indices

    def __call__(self, feat):
        if self.metric == 'cosine':
            feat = normalize(feat)

        if self.protocol == 'BLUFR':
            if self.metric == 'cosine':
                score_mat = np.dot(feat,feat.T)
            elif self.metric == 'Euclidean':
                score_mat = np.zeros((feat.shape[0],feat.shape[0]))
                for i in range(feat.shape[0]):
                    temp=feat[i,:]
                    temp=temp[None,:]
                    temp1=np.sum(np.square(feat-temp),axis=1)
                    score_mat[i,:] = -1*temp1[:]
            else:
                raise(RuntimeError('Metric doest not support!'))
            score_vec,label_vec = get_pairwise_score_label(score_mat,self.label)
            TARs,FARs,thresholds = ROC(score_vec,label_vec)

        elif self.protocol == 'LFW' or self.protocol == 'RFW_one_race':
            feat1 = feat[self.pair_indices[:,0]]
            feat2 = feat[self.pair_indices[:,1]]
            if self.metric == 'cosine':
                score_vec = np.sum(feat1*feat2, axis=1)
            elif self.metric == 'Euclidean':
                score_vec = -1*np.sum(np.square(feat1 - feat2), axis=1)
            else:
                raise(RuntimeError('The disctnace metric does not support!'))
            avg,std,thd = cross_valid_accuracy(score_vec, self.issame_label,
                self.pair_indices, self.nfolds)
            print("Accuracy is {}".format(avg))
            return avg,std,thd

        elif self.protocol == 'RFW':
            acc_dict = {}
            keys = list(self.issame_label)
            for key in keys:
                feat1 = feat[self.pair_indices[key][:,0]]
                feat2 = feat[self.pair_indices[key][:,1]]
                score_vec = np.sum(feat1*feat2, axis=1)
                acc,thd = accuracy(score_vec, self.issame_label[key])
                print("Accuracy of {} is {}".format(key, acc))
                acc_dict[key] = acc
            avg = np.mean([100.0*acc_dict[x] for x in keys])
            std = np.std([100.0*acc_dict[x] for x in keys])
            return avg, std, acc_dict

        elif self.protocol == 'CFP':
            feat_ori = np.load('/scratch/gongsixue/face_resolution/feats/feat_cfp_112x112.npz')
            feat_ori = feat_ori['feat']
            feat_ori = normalize(feat_ori)
            
            folds = ['01','02','03','04','05','06','07','08','09','10']
            accs = []
            for i in range(10):
                splitfolder = os.path.join(self.pair_index_filename, folds[i])
                with open(os.path.join(splitfolder, 'diff.txt')) as f:
                    lines = f.readlines()
                    pair1 = [int(x.rstrip('\n').split(',')[0])-1 for x in lines]
                    pair2 = [int(x.rstrip('\n').split(',')[1])-1 for x in lines]
                    labels = np.zeros((len(lines)))
                with open(os.path.join(splitfolder, 'same.txt')) as f:
                    lines = f.readlines()
                    pair1.extend([int(x.rstrip('\n').split(',')[0])-1 for x in lines])
                    pair2.extend([int(x.rstrip('\n').split(',')[1])-1 for x in lines])
                    labels = np.concatenate((labels,np.ones((len(lines)))),axis=0)
                label_vec = labels.astype('bool')
                pair1 = np.array(pair1).astype('int')
                pair2 = np.array(pair2).astype('int')
                feat1 = feat[pair1,:]
                feat2 = feat_ori[pair2,:]
                if self.metric == 'cosine':
                    score_vec = np.sum(feat1*feat2, axis=1)
                elif self.metric == 'Euclidean':
                    score_vec = -1*np.sum(np.square(feat1 - feat2), axis=1)
                else:
                    raise(RuntimeError('The disctnace metric does not support!'))
                acc,thd = accuracy(score_vec, label_vec)
                accs.append(acc)
            avg = np.mean(accs)
            std = np.std(accs)
            print("Accuracy is {}, STD is {}".format(avg,std))
            return avg,std
        else:
            raise(RuntimeError('Protocol doest not support!'))
        
        tar = find_tar(FARs, TARs, 0.01)
        print("TAR is {} at FAR 0.1%".format(tar))

        return TARs,FARs,thresholds

def find_tar(FAR, TAR, far):
    i = 0
    while FAR[i] < far:
        i += 1
    tar = TAR[i]
    return tar

def ROC(score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False):
    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    FARs = get_FARs()
    if thresholds is None:
        thresholds = find_thresholds_by_FAR(score_vec, label_vec, FARs=FARs)
    
    assert len(thresholds.shape)==1 
    if np.size(thresholds) > 10000:
        print('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))

    score_pos = score_vec[label_vec]
    score_neg = score_vec[~label_vec]
    index_vec = np.arange(len(score_vec))
    index_pos = index_vec[label_vec]
    index_neg = index_vec[~label_vec]
    num_pos = len(score_pos)
    num_neg = len(score_neg)

    # FARs would be check again
    TARs = np.zeros(thresholds.shape[0])
    FARs = np.zeros(thresholds.shape[0])
    false_accept_indices = []
    false_reject_indices = []
    for i,threshold in enumerate(thresholds):
        correct_pos = score_pos >= threshold
        correct_neg = score_neg < threshold
        TARs[i] = np.mean(correct_pos)
        FARs[i] = np.mean(~correct_neg)
        if get_false_indices:
            false_accept_indices.append(index_pos[~correct_pos])
            false_reject_indices.append(index_neg[~correct_neg])

    if get_false_indices:
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds

def cross_valid_accuracy(score_vec, label_vec, indices, nfolds):
    kfold = KFold(n_splits=nfolds, shuffle=False)

    accuracies = np.zeros(nfolds)
    thresholds = np.zeros(nfolds)

    for fold_idx, (train_set, test_set) in enumerate(kfold.split(indices)):
        _,threshold = accuracy(score_vec[train_set],label_vec[train_set])
        acc,_ = accuracy(score_vec[test_set], label_vec[test_set], threshold)
        accuracies[fold_idx] = acc
        thresholds[fold_idx] = threshold

    avg = np.mean(accuracies)
    std = np.std(accuracies)
    thd = np.mean(thresholds)

    return avg,std,thd

def score_per_pair(metric,feat,template,pair):
    label = pair[0]
    index = pair[1]
    if template is None:
        feat1 = feat[index[0]].view(1,-1)
        feat2 = feat[index[1]].view(1,-1)

    else:
        temp1 = index.split(',')[0]
        temp2 = index.split(',')[1]
        if temp1 in template.keys() and temp2 in template.keys():
            idA = template[temp1][1]
            feat1 = np.mean(feat[idA,:],axis=0,keepdims=True)
            idB = template[temp2][1]
            feat2 = np.mean(feat[idB,:],axis=0,keepdims=True)
        else:
            feat1 = None
            feat2 = None
    
    if feat1 is not None and feat2 is not None:
        if metric == 'cosine':
            score = np.dot(feat1,feat2.T)
        elif metric == 'Euclidean':
            score = -1*np.sum(np.square(feat1-feat2))
        else:
            raise(RuntimeError('The disctnace metric does not support!'))
        return label,score
    else:
        return label,None

def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=10e-8):
    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        epsilon = 10e-5
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = (num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds

def accuracy(score_vec, label_vec, threshold=None):
    assert len(score_vec.shape)==1
    assert len(label_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype==np.bool
    
    # find thresholds by TAR
    if threshold is None:
        score_pos = score_vec[label_vec==True]
        thresholds = np.sort(score_pos)[::1]    
        if np.size(thresholds) > 10000:
            print('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))    
        # Loop Computation
        accuracies = np.zeros(np.size(thresholds))
        for i, threshold in enumerate(thresholds):
            pred_vec = score_vec>=threshold
            accuracies[i] = np.mean(pred_vec==label_vec)
        # Matrix Computation, Each column is a threshold
        argmax = np.argmax(accuracies)
        accuracy = accuracies[argmax]
        threshold = np.mean(thresholds[accuracies==accuracy])
    else:
        pred_vec = score_vec>=threshold
        accuracy = np.mean(pred_vec==label_vec)

    return accuracy, threshold
