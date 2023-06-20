# implemented vgg16 in Detectron2

Mainly based on code from https://github.com/facebookresearch/detectron2/pull/1584, [@ashnair1](https://github.com/ashnair1)

and https://github.com/facebookresearch/detectron2

After removing maxpool layer at the bottom of vgg backbone, there are 3 main modifications I made to fit my own settings:

* Remove the bn layers and set the bias=True in every conv layer of VGG block
* Add gradient clipping layers following various other implementations, such as [detectron](https://github.com/facebookresearch/Detectron) and [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).
* **Add box_head layers of 2 fc layers following several implementations such as [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)**, which is the key to a normal performance. Specifically, I used the second (4096, 4096) and third (512x7x7, 4096) fc layers bottom up of the classifier layers of vgg from torchvision.models, following [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), [DA_detection](https://github.com/VisionLearningGroup/DA_Detection), etc. I also modified code from [@Shuntw6096](https://github.com/Shuntw6096) to convert the classifier params of vgg to box_head params.

**After the modification, the results should match most experiment results from previous papers using vgg backboned faster rcnn, such as [Multi-Source Domain Adaptation for Object Detection](https://arxiv.org/pdf/2106.15793.pdf) and [Strong-Weak Distribution Alignment for Adaptive Object Detection](https://arxiv.org/pdf/1812.04798.pdf). 
By the way, I've tested using StandardROIHeads. As [@ashnair1](https://github.com/ashnair1) in [Add VGG backbones by ashnair1](https://github.com/facebookresearch/detectron2/pull/1584) pointed out and the performance is indeed very poor.**




<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

<a href="https://opensource.facebook.com/support-ukraine">
  <img src="https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB" alt="Support Ukraine - Help Provide Humanitarian Aid to Ukraine." />
</a>

Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>
<br>

## Learn More about Detectron2

Explain Like I’m 5: Detectron2            |  Using Machine Learning with Detectron2
:-------------------------:|:-------------------------:
[![Explain Like I’m 5: Detectron2](https://img.youtube.com/vi/1oq1Ye7dFqc/0.jpg)](https://www.youtube.com/watch?v=1oq1Ye7dFqc)  |  [![Using Machine Learning with Detectron2](https://img.youtube.com/vi/eUSgtfK4ivk/0.jpg)](https://www.youtube.com/watch?v=eUSgtfK4ivk)

## What's New
* Includes new capabilities such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend,
  DeepLab, ViTDet, MViTv2 etc.
* Used as a library to support building [research projects](projects/) on top of it.
* Models can be exported to TorchScript format or Caffe2 format for deployment.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Getting Started

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),
and the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn about basic usage.

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
