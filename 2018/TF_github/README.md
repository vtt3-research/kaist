# Transfer Learning
KAIST Transfer Learning for VTT


# Pytorch Implementation of Domain Adaptive Faster R-CNN
Pytorch version of [da-faster-rcnn](https://arxiv.org/abs/1803.03243).

This repository is based on [Pytorch Implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch) and original [da-faster-rcnn code](https://github.com/yuhuayc/da-faster-rcnn).

Requirements
------------

* Python 2.7 or 3.x
* Pytorch 0.4.0 (not support 0.4.1 or higher)

Preparation
--------
Example of adapting from **Cityscapes** to **Foggy Cityscapes**.

Source: train set of Cityscapes

Target: val set of Foggy Cityscapes

####Prepare dataset
1. Download **gtFine_trainvaltest.zip**, **leftImg8bit_trainvaltest.zip** and **leftImg8bit_trainvaltest_foggy.zip** from [here](www.cityscapes-dataset.com)
2. Prepare the data using the scripts 'data/prepare_data.m'

####Pretrained model
1. Download pretrained model [here](https://github.com/jwyang/faster-rcnn.pytorch#pretrained-model).
2. Put them into 'data/pretrained_model/'.

####Compile the cuda dependencies.
	
	pip install -r requirements.txt
	cd lib
	sh make.sh

Train
------
	python trainval_net_da_only_im.py --cuda


Validation
-------
	python test_net_da_only_im.py --cuda


Results
------
To be updated

Citations
----------
	@inproceedings{chen2018domain,
	  title={Domain Adaptive Faster R-CNN for Object Detection in the Wild},
	  author={Chen, Yuhua and Li, Wen and Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
	  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
	  year={2018}
	}
	
	@article{jjfaster2rcnn,
	    Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
	    Title = {A Faster Pytorch Implementation of Faster R-CNN},
	    Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
	    Year = {2017}
	}
	
	@inproceedings{renNIPS15fasterrcnn,
	    Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
	    Title = {Faster {R-CNN}: Towards Real-Time Object Detection
	             with Region Proposal Networks},
	    Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
	    Year = {2015}
	}

