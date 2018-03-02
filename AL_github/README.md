# Active Learning
KAIST Active Learning for VTT

## Requirements

* Python2.7
* Pytorch

## Training

#### Download dataset
* Caltech256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>

#### Training

train.py supposes that your data folder includes 'labeled', 'unlabeled', 'test' folders.
In train.py, set arg.data as the directory of your data folder
```
python train.py
```

## Testing

Test will be done during and after training

## Result

To be updated

## References

[1] K. Wang D. Zhang Y. Li R. Zhang L. Lin "Cost-effective active learning for deep image classification", IEEE Trans. Circuits Syst. Video Technol. 2016. 