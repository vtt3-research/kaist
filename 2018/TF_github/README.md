# Transfer Learning
KAIST Transfer Learning for VTT


# Pytorch Implementation of Domain Adaptive Faster R-CNN
Pytorch version of [da-faster-rcnn](https://arxiv.org/abs/1803.03243).

The original code with Caffe can be found [here](https://github.com/yuhuayc/da-faster-rcnn).

This repository is based on [Pytorch Implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch).

Requirements
------------

* Python2.7 or 3.x
* Pytorch >= 0.3.0
* ``` pip install -r requirements.txt ```

Prepare Dataset
--------
Adapting from **Cityscapes** to **Foggy Cityscapes**.

Source: train set of Cityscapes

Target: val set of Foggy Cityscapes

1. Download **gtFine_trainvaltest.zip**, **leftImg8bit_trainvaltest.zip** and **leftImg8bit_trainvaltest_foggy.zip** from [here](www.cityscapes-dataset.com)
2. Prepare the data using the scripts in 'data/prepare_data.m'
3. Structure of the prepared dataset is same with Pascal VOC.

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

