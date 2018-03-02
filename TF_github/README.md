# Transfer Learning
KAIST Transfer Learning for VTT


Requirements
------------

* Python2.7
* Pytorch

Training
--------

#### Download dataset
* Office-31 <https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code>

#### Training
```
python main.py --epochs [epochs_num] -b [batch_num] --lr [lr_num] -s [split_num]  -nc [class_num] --print-freq [freq] --pretrained [source root] [target root]

```

Testing
-------

Test will be done during and after training

Result
------
To be updated

References
----------