# *GAC*: Mitigating Face Recognition Bias via Group Adaptive Classifier

By Sixue Gong, Xiaoming Liu, and Anil K. Jain

## Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Usage](#usage)

### Introduction

This code archive includes the Python implementation of group adaptive classifier for demographic bias mitigation -- GAC. Our work, *GAC*, mitigates bias by using adaptive convolution kernels and attention mechanisms on faces based on their demographic attributes. The adaptive module comprises kernel masks and channel-wise attention maps for each demographic group so as to activate different facial regions for identification, leading to more discriminative features pertinent to their demographics.

### Citation

If you think **GAC** is useful to your research, please cite:

    @inproceedings{gong2021mitigating,
        title={Mitigating Face Recognition Bias via Group Adaptive Classifier},
        author={Gong, Sixue and Liu, Xiaoming and Jain, Anil K},
        booktitle={CVPR},
        year={2021}
    }
    
**Link** to the paper: https://arxiv.org/abs/1803.09672

### Requirements

1. Require `Python3`
2. Require `PyTorch1.6`
3. Require `TensorBoard`
4. Check `Requirements.txt` for detailed dependencies.

### Usage

# How to use it?
It is easy to be customized to fulfill your own demands. All you need to do is to create the necessary classes which I list below:
 1. **Data Loader**: defines how you load your training and testing data.
 For your information, I put some data loaders I often use in the "datasets" folder, such as "folderlist.py" (loading images in a given folder), "filelist.py" (loading images in a file list written in a text file), "triplet.py" (loading images as triplets), and so on.
 2. **Network Architecture**: uses a nn.Module class to define your neural network.
 I put some well-kown networks in face recognition domain in the "models" folder, such as "resnet.py" (ResNet), "insightface.py" (ArcFace), and "sphereface.py" (SphereFace); "gac_irse.py" and "gac_resnet.py" (GAC with different backbone network).
 3. **Loss Method**: Pytorch provides some comman loss functions in torch.nn library, for example CrossEntropyLoss or MSELoss. And you can also design your own loss function by using a nn.Module class, which is similary to writting a network architecture.
 Still, there are some loss functions I wrote for face recognition and shoe image retrieval, that can be found in "losses" folder.
 4. **Evaluation Metric**: a class to measure the accuracy (performance) of your network model, i.e., how to test your network.
 Again, I show examples in the "evaluate" folder.

## More remarks
Apart from the four major classes I just mensioned before, you may need to edit the belowing files as well, to make the whole program work. All these files are in the root folder.
1. "config.py" and "args.py". These two files help to load parameters of the configuration. The former defines what kinds of parameters in the configuration, while the latter assigns the specific value to each parameter. You can assign values to any parameter in the configuration except "--result_path", where the trained models and log files will be saved. This can only be set by comman line, and you can look at the example, "train.sh" for more information.
2. "dataloader.py". You may need to add your own loader class to this file.
3. "train_cls.py" and "test_cls.py". You may need to change the parts of data loading, loss computation, and performance evaluation in these two files, to make them compatible with the data loader, loss method, and evaluation metric that you define.

# Storage
Once the network starts training, each currently best model will be saved so that you can stop it and then resume any time you want. The function about models and log files saving can be found in "plugins" folder.
1. **Monitor**: print loss and accuracy on the screen.
2. **Logger**:  write the loss and accuracy to log files.

# Visualization
I use a TensorBoard, to visualize the training process.
