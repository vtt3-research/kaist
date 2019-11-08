# Diversify and Match 

### Acknowledgment

The implementation is built on the pytorch implementation of Faster RCNN [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)


## Preparation
1. Clone the code and create a folder
```
git clone https://github.com/TKKim93/DivMatch.git
cd faster-rcnn.pytorch && mkdir data
```

2. Build the Cython modules
```Shell
cd DivMatch/lib
sh make.sh
``` 

### Prerequisites

* Python 3.6
* Pytorch 0.4.0 or 0.4.1
* CUDA 8.0 or higher
* cython, cffi, opencv-python, scipy, easydict, matplotlib, pyyaml

### Pretrained Model
You can download pretrained VGG and ResNet101 from [jwyang's repository](https://github.com/jwyang/faster-rcnn.pytorch). Default location in my code is './data/pretrained_model/'.

### Repository Structure
```
DivMatch
├── cfgs
├── data
│   ├── pretrained_model
├── datasets
│   ├── clipart
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   ├── watercolor
│   ├── comic
│   ├── Pascal
├── lib
├── models (save location)
```

## Example
### Diversification stage
Here are the simplest ways to generate shifted domains via [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Some of them performs unnecessary computations, thus you may revise the I2I code for faster training.
1. CP shift

Change line 177 in models/cycle_gan_model.py to
```
loss_G = self.loss_G_A + self.loss_G_B + self.loss_idt_A + self.loss_idt_B
```
2. R shift

Change line 177 in models/cycle_gan_model.py to
```
loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
```
3. CPR shift

Use the original cyclegan model.

### Matching stage
Here is an example of adapting from Pascal VOC to Clipart1k:
1. You can prepare the Pascal VOC datasets from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and the Clipart1k dataset from [cross-domain-detection](https://github.com/naoto0804/cross-domain-detection) in VOC data format.
2. Shift the source domain through domain shifter. Basically, I used a residual generator and a patchGAN discriminator. For the short cut, you can download some examples of shifted domains (Link) and put these datasets into data folder.
3. Train the object detector through MRL for the Pascal -> Clipart1k adaptation task.
```
    python train.py --dataset clipart --net vgg16 --cuda
```
4. Test the model
```
    python test.py --dataset clipart --net vgg16 --cuda
```
