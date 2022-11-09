# Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/salient-object-detection-on-hku-is)](https://paperswithcode.com/sota/salient-object-detection-on-hku-is?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/salient-object-detection-on-duts-te)](https://paperswithcode.com/sota/salient-object-detection-on-duts-te?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/salient-object-detection-on-pascal-s)](https://paperswithcode.com/sota/salient-object-detection-on-pascal-s?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/salient-object-detection-on-ecssd)](https://paperswithcode.com/sota/salient-object-detection-on-ecssd?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/salient-object-detection-on-dut-omron)](https://paperswithcode.com/sota/salient-object-detection-on-dut-omron?p=revisiting-image-pyramid-structure-for-high)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/rgb-salient-object-detection-on-davis-s)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-davis-s?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/rgb-salient-object-detection-on-hrsod)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-hrsod?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/rgb-salient-object-detection-on-uhrsd)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-uhrsd?p=revisiting-image-pyramid-structure-for-high)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/dichotomous-image-segmentation-on-dis-vd)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-vd?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/dichotomous-image-segmentation-on-dis-te1)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te1?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/dichotomous-image-segmentation-on-dis-te2)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te2?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/dichotomous-image-segmentation-on-dis-te3)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te3?p=revisiting-image-pyramid-structure-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-image-pyramid-structure-for-high/dichotomous-image-segmentation-on-dis-te4)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te4?p=revisiting-image-pyramid-structure-for-high)


Official PyTorch implementation of PyTorch implementation of Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)

To appear in the 16th Asian Conference on Computer Vision (ACCV2022)

<p align="center">
<a href="https://github.com/plemeri/InSPyReNet/blob/main/LICENSE"><img  src="https://img.shields.io/badge/license-MIT-blue"></a>
<a href="https://arxiv.org/abs/2209.09475"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
<a href=""><img  src="https://img.shields.io/static/v1?label=inproceedings&message=Paper&color=orange"></a>
<a href="https://huggingface.co/spaces/taskswithcode/salient-object-detection"><img  src="https://img.shields.io/static/v1?label=HuggingFace&message=Demo&color=yellow"></a>
<a href="https://www.taskswithcode.com/salient_object_detection/"><img  src="https://img.shields.io/static/v1?label=TasksWithCode&message=Demo&color=blue"></a>
<a href="https://colab.research.google.com/github/taskswithcode/InSPyReNet/blob/main/TWCSOD.ipynb"><img  src="https://img.shields.io/static/v1?label=Colab&message=Demo&color=blue"></a>
</p>

> **Abstract:**
  Salient object detection (SOD) has been in the spotlight recently, yet has been studied less for high-resolution (HR) images. 
  Unfortunately, HR images and their pixel-level annotations are certainly more labor-intensive and time-consuming compared to low-resolution (LR) images.
  Therefore, we propose an image pyramid-based SOD framework, Inverse Saliency Pyramid Reconstruction Network (InSPyReNet), for HR prediction without any of HR datasets.
  We design InSPyReNet to produce a strict image pyramid structure of saliency map, which enables to ensemble multiple results with pyramid-based image blending.
  For HR prediction, we design a pyramid blending method which synthesizes two different image pyramids from a pair of LR and HR scale from the same image to overcome effective receptive field (ERF) discrepancy. Our extensive evaluation on public LR and HR SOD benchmarks demonstrates that InSPyReNet surpasses the State-of-the-Art (SotA) methods on various SOD metrics and boundary accuracy.
  
## News :newspaper:

[2022.10.04] [TasksWithCode](https://github.com/taskswithcode) mentioned our work in [Blog](https://medium.com/@taskswithcode/twc-9-7c960c921f69) and reproducing our work on [Colab](https://github.com/taskswithcode/InSPyReNet). Thank you for your attention!

[2022.10.20] :new: We trained our model on [Dichotomous Image Segmentation dataset (DIS5K)](https://xuebinqin.github.io/dis/index.html) and showed competitive results! Trained checkpoint and pre-computed segmentation masks are available in [Model Zoo](./docs/model_zoo.md)). Also, you can check our qualitative and quantitative results in [Results](#results) section.

[2022.10.28] Multi GPU training for latest pytorch is now available.

[2022.10.31] :new: [TasksWithCode](https://github.com/taskswithcode) provided an amazing web demo with [HuggingFace](https://huggingface.co). Visit the [WepApp](https://huggingface.co/spaces/taskswithcode/salient-object-detection) and try with your image! 

## Demo :rocket:

* <img src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg height="20px" width="20px"> Try [WepApp](https://huggingface.co/spaces/taskswithcode/salient-object-detection) on HuggingFace to generate your own results!

[WepApp](https://huggingface.co/spaces/taskswithcode/salient-object-detection) | [Image Sample](./figures/demo_image.gif) | [Video Sample](./figures/demo_video.gif)
:-:|:-:|:-:
<img src=./figures/demo_webapp.gif height="250px" width="250px"> | <img src=./figures/demo_image.gif height="250px" width="250px"> | <img src=./figures/demo_video.gif height="250px" width="250px">

## Architecture

[InSPyReNet](./figures/fig_architecture.png) | [pyramid blending](./figures/fig_pyramid_blending.png)
:-:|:-:
<img src=./figures/fig_architecture.png height="350px" width="350px"> | <img src=./figures/fig_pyramid_blending.png height="350px" width="350px">

## Easy Download

<details><summary>How to use easy download</summary>
<p>

Downloading each dataset, checkpoint is quite bothering, even for me :zzz:. Instead, you can download data we provide including `ImageNet pre-trained backbone checkpoints`, `Training Datasets`, `Testing Datasets for benchmark`, `Pre-trained model checkpoints`, `Pre-computed saliency maps` with single command below.
```
python utils/download.py --extra --dest [DEST]
```

* `--extra, -e`: Without this argument, only the datasets, checkpoint, and results from our main paper will be downloaded. Otherwise, all data will be downloaded including results from supplementary material and DIS5K results.
* `--dest [DEST], -d [DEST]`: If you want to specify the destination, use this argument. It will automatically create a symbolic links of the destination folders inside `data` and `snapshots`. Use this argument if you want to download data on other physical disk. Otherwise, it will download inside this repository folder.

</p>
</details>

## Getting Started

Please refer to [getting_started.md](./docs/getting_started.md) for trainig, testing, evaluating, and inferencing on your own images.

## Model Zoo

Please refer to [model_zoo.md](./docs/model_zoo.md) for downloading pre-trained models and pre-computed saliency maps.

## Results

* Quantitative

[LR Benchmark](./figures/fig_quantitative.png) | [HR Benchmark](./figures/fig_quantitative2.png) | [HR Benchmark (Trained with extra DB)](./figures/fig_quantitative3.png) 
:-:|:-:|:-:
<img src=./figures/fig_quantitative.png height="250px" width="250px"> | <img src=./figures/fig_quantitative2.png height="250px" width="250px"> | <img src=./figures/fig_quantitative3.png height="250px" width="250px">

* :new: [Added in 2022.10.20] Quantitative results on DIS5K dataset [[Log file](https://postechackr-my.sharepoint.com/:t:/g/personal/taehoon1018_postech_ac_kr/EeczZ1XEboZKhxqif9m1VwsBhMc--dLYqlZ_5TicEXr2ZA?e=aCFXhp)]
  * *: HCE here is relax Human Correction Error which is proposed in DIS([project page](https://xuebinqin.github.io/dis/index.html) | [paper](https://arxiv.org/pdf/2203.03041.pdf)) and you can compute yourself from their [github](https://github.com/xuebinqin/DIS) repository.

Dataset | Sm | mae | adpEm | maxEm | avgEm | adpFm | maxFm | avgFm | wFm | mBA | HCE<sup>*</sup>
:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|
 DIS-VD  | 0.8868 | 0.0427 | 0.9145 | 0.9352 | 0.9217 | 0.8295 | 0.8760 | 0.8523 | 0.8259 | 0.7654 | 905
 DIS-TE1 | 0.8618 | 0.0447 | 0.8679 | 0.9071 | 0.8952 | 0.7556 | 0.8341 | 0.8083 | 0.7771 | 0.7453 | 148
 DIS-TE2 | 0.8934 | 0.0383 | 0.9131 | 0.9356 | 0.9253 | 0.8281 | 0.8811 | 0.8599 | 0.8339 | 0.7587 | 316
 DIS-TE3 | 0.9019 | 0.0381 | 0.9278 | 0.9496 | 0.9376 | 0.8529 | 0.9038 | 0.8802 | 0.8558 | 0.7741 | 582
 DIS-TE4 | 0.8913 | 0.0461 | 0.9316 | 0.9433 | 0.9255 | 0.8545 | 0.8915 | 0.8655 | 0.8395 | 0.7789 | 2243

* Qualitative

[DAVIS-S & HRSOD](./figures/fig_qualitative.png) | [UHRSD](./figures/fig_qualitative2.png) | [UHRSD (Trained with extra DB)](./figures/fig_qualitative3.jpg) | :new: [DIS](./figures/fig_qualitative_dis.png)
:-:|:-:|:-:|:-:
<img src=./figures/fig_qualitative.png height="250px" width="250px"> | <img src=./figures/fig_qualitative2.png height="250px" width="250px"> | <img src=./figures/fig_qualitative3.jpg height="250px" width="250px"> | <img src=./figures/fig_qualitative_dis.png height="250px" width="250px">

## Citation

```
@inproceedings{kim2022revisiting,
  title={Revisiting Image Pyramid Structure for High Resolution Salient Object Detection},
  author={Kim, Taehun and Kim, Kunhee and Lee, Joonyeong and Cha, Dongmin and Lee, Jiho and Kim, Daijin},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2022}
}
```

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) 
(No.2017-0-00897, Development of Object Detection and Recognition for Intelligent Vehicles) and 
(No.B0101-15-0266, Development of High Performance Visual BigData Discovery Platform for Large-Scale Realtime Data Analysis)

### Special Thanks to
* [TasksWithCode](https://github.com/taskswithcode) team for sharing our work and amazing web app demo.


## References

### Related Works

* Towards High-Resolution Salient Object Detection ([paper](https://drive.google.com/open?id=15o-Fel0BSyNulGoptrxfHR0t22qMHlTr) | [github](https://github.com/yi94code/HRSOD))
* Disentangled High Quality Salient Object Detection ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_Disentangled_High_Quality_Salient_Object_Detection_ICCV_2021_paper.pdf) | [github](https://github.com/luckybird1994/HQSOD))
* Pyramid Grafting Network for One-Stage High Resolution Saliency Detection ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Pyramid_Grafting_Network_for_One-Stage_High_Resolution_Saliency_Detection_CVPR_2022_paper.pdf) | [github](https://github.com/iCVTEAM/PGNet))

### Resources

* Backbones: [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels), [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

* Datasets:
  * LR Benchmarks: [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [ECSSD](https://i.cs.hku.hk/~gbli/deep_saliency.html), [HKU-IS](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [PASCAL-S](http://cbi.gatech.edu/salobj/)
  * HR Benchmarks: [DAVIS-S, HRSOD](https://github.com/yi94code/HRSOD), [UHRSD](https://github.com/iCVTEAM/PGNet)
  * Dichotomous Image Segmentation: [DIS5K](https://xuebinqin.github.io/dis/index.html)

* Evaluation Toolkit
  * SOD Metrics (e.g., S-measure): [PySOD Metrics](https://github.com/lartpang/PySODMetrics)
  * Boundary Metric (mBA): [CascadePSP](https://github.com/hkchengrex/CascadePSP)
