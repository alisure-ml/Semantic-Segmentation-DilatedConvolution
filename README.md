# FCN-DilatedConvolution

A *Tensorflow* implementation of semantic segmentation according to 
[Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) by Yu and Koltun.

Pretrained weights have been converted to TensorFlow from the [original Caffe implementation](https://github.com/fyu/dilation).


## Run
1. Download pretrained weights from here:
    * [CityScapes weights](https://drive.google.com/open?id=0Bx9YaGcDPu3XR0d4cXVSWmtVdEE)
    * [CamVid weights](https://drive.google.com/open?id=0Bx9YaGcDPu3Xd0JrcXZpTEpkb0U)

2. Move weights file into [`pretrained`](pretrained) directory.

3. Run the model on the test image by executing [`InferenceRunner.py`](InferenceRunner.py).


## Reference
* [Multi Scale Context Aggregation by Dilated Convolutions](https://github.com/alisure-ml/FCN-Review/blob/master/Multi_Scale_Context_Aggregation_by_Dilated_Convolutions.md)
* [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
* [ndrplz/dilation-tensorflow](https://github.com/ndrplz/dilation-tensorflow)
