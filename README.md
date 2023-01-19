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
<a href="https://openaccess.thecvf.com/content/ACCV2022/html/Kim_Revisiting_Image_Pyramid_Structure_for_High_Resolution_Salient_Object_Detection_ACCV_2022_paper.html"><img  src="https://img.shields.io/static/v1?label=inproceedings&message=Paper&color=orange"></a>
<a href="https://huggingface.co/spaces/taskswithcode/salient-object-detection"><img  src="https://img.shields.io/static/v1?label=HuggingFace&message=Demo&color=yellow"></a>
<a href="https://www.taskswithcode.com/salient_object_detection/"><img  src="https://img.shields.io/static/v1?label=TasksWithCode&message=Demo&color=blue"></a>
<a href="https://colab.research.google.com/github/taskswithcode/InSPyReNet/blob/main/TWCSOD.ipynb"><img  src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

> [Taehun Kim](https://scholar.google.co.kr/citations?user=f12-9yQAAAAJ&hl=en), [Kunhee Kim](https://scholar.google.co.kr/citations?user=6sU5r7MAAAAJ&hl=en), [Joonyeong Lee](https://scholar.google.co.kr/citations?hl=en&user=pOM4zSYAAAAJ), Dongmin Cha, [Jiho Lee](https://scholar.google.co.kr/citations?user=1Q1awj8AAAAJ&hl=en), [Daijin Kim](https://scholar.google.co.kr/citations?user=Mw6anjAAAAAJ&hl=en)

> **Abstract:**
  Salient object detection (SOD) has been in the spotlight recently, yet has been studied less for high-resolution (HR) images. 
  Unfortunately, HR images and their pixel-level annotations are certainly more labor-intensive and time-consuming compared to low-resolution (LR) images.
  Therefore, we propose an image pyramid-based SOD framework, Inverse Saliency Pyramid Reconstruction Network (InSPyReNet), for HR prediction without any of HR datasets.
  We design InSPyReNet to produce a strict image pyramid structure of saliency map, which enables to ensemble multiple results with pyramid-based image blending.
  For HR prediction, we design a pyramid blending method which synthesizes two different image pyramids from a pair of LR and HR scale from the same image to overcome effective receptive field (ERF) discrepancy. Our extensive evaluation on public LR and HR SOD benchmarks demonstrates that InSPyReNet surpasses the State-of-the-Art (SotA) methods on various SOD metrics and boundary accuracy.

## Contents

1. [News](#news-newspaper)
2. [Demo](#demo-rocket)
3. [Applications](#applications-video_game)
4. [Easy Download](#easy-download-cake)
5. [Getting Started](#getting-started-flight_departure)
6. [Model Zoo](#model-zoo-giraffe)
7. [Results](#results-100)
    * [Quantitative Results](#quantitative-results)
    * [Qualitative Results](#qualitative-results)
8. [Citation](#citation)
9. [Acknowledgement](#acknowledgement)
    * [Special Thanks to](#special-thanks-to-tada)
10. [References](#references)
  
## News :newspaper:

[2022.10.04] [TasksWithCode](https://github.com/taskswithcode) mentioned our work in [Blog](https://medium.com/@taskswithcode/twc-9-7c960c921f69) and reproducing our work on [Colab](https://github.com/taskswithcode/InSPyReNet). Thank you for your attention!

[2022.10.20] We trained our model on [Dichotomous Image Segmentation dataset (DIS5K)](https://xuebinqin.github.io/dis/index.html) and showed competitive results! Trained checkpoint and pre-computed segmentation masks are available in [Model Zoo](./docs/model_zoo.md)). Also, you can check our qualitative and quantitative results in [Results](#100-results) section.

[2022.10.28] Multi GPU training for latest pytorch is now available.

[2022.10.31] [TasksWithCode](https://github.com/taskswithcode) provided an amazing web demo with [HuggingFace](https://huggingface.co). Visit the [WepApp](https://huggingface.co/spaces/taskswithcode/salient-object-detection) and try with your image! 

[2022.11.09] :car: Lane segmentation for driving scene built based on InSPyReNet is available in [`LaneSOD`](https://github.com/plemeri/LaneSOD) repository.

[2022.11.18] I am speaking at The 16th Asian Conference on Computer Vision (ACCV2022). Please check out my talk if you're attending the event! #ACCV2022 #Macau - via #Whova event app

[2022.11.23] We made our work available on pypi package. Please visit [`transparent-background`](https://github.com/plemeri/transparent-background) to download our tool and try on your machine. It works as command-line tool and python API.

[2023.01.18] [rsreetech](https://github.com/rsreetech) shared a tutorial for our pypi package [`transparent-background`](https://github.com/plemeri/transparent-background) using colab. :tv: [[Youtube](https://www.youtube.com/watch?v=jKuQEnKmv4A)]

## Demo :rocket:

[Image Sample](./figures/demo_image.gif) | [Video Sample](./figures/demo_video.gif)
:-:|:-:
<img src=./figures/demo_image.gif height=200px> | <img src=./figures/demo_video.gif height=200px>

## Applications :video_game: 
Here are some applications/extensions of our work.
### Web Application <img src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg height="20px" width="20px"> 
[TasksWithCode](https://github.com/taskswithcode) provided [WepApp](https://huggingface.co/spaces/taskswithcode/salient-object-detection) on HuggingFace to generate your own results!

[Web Demo](https://huggingface.co/spaces/taskswithcode/salient-object-detection) |
|:-:
<img src=./figures/demo_webapp.gif height=200px> |

### Command-line Tool / Python API :pager: 
Try using our model as command-line tool or python API. More details about how to use is available on [`transparent-background`](https://github.com/plemeri/transparent-background).
```bash
pip install transparent-background
```
### Lane Segmentation :car: 
We extend our model to detect lane markers in a driving scene in [`LaneSOD`](https://github.com/plemeri/LaneSOD)

[Lane Segmentation](https://github.com/plemeri/LaneSOD) |
|:-:
<img src=https://github.com/plemeri/LaneSOD/blob/main/figures/Teaser.gif height=200px> |

## Easy Download :cake: 

<details><summary>How to use easy download</summary>
<p>

Downloading each dataset, checkpoint is quite bothering, even for me :zzz:. Instead, you can download data we provide including `ImageNet pre-trained backbone checkpoints`, `Training Datasets`, `Testing Datasets for benchmark`, `Pre-trained model checkpoints`, `Pre-computed saliency maps` with single command below.
```bash
python utils/download.py --extra --dest [DEST]
```

* `--extra, -e`: Without this argument, only the datasets, checkpoint, and results from our main paper will be downloaded. Otherwise, all data will be downloaded including results from supplementary material and DIS5K results.
* `--dest [DEST], -d [DEST]`: If you want to specify the destination, use this argument. It will automatically create a symbolic links of the destination folders inside `data` and `snapshots`. Use this argument if you want to download data on other physical disk. Otherwise, it will download inside this repository folder.

If you want to download a certain checkpoint or pre-computed map, please refer to [Getting Started](#flight_departure-getting-started) and [Model Zoo](#giraffe-model-zoo).

</p>
</details>

## Getting Started :flight_departure:

Please refer to [getting_started.md](./docs/getting_started.md) for training, testing, and evaluating on benchmarks, and inferencing on your own images.

## Model Zoo :giraffe:

Please refer to [model_zoo.md](./docs/model_zoo.md) for downloading pre-trained models and pre-computed saliency maps.

## Results :100:

### Quantitative Results

[LR Benchmark](./figures/fig_quantitative.png) | [HR Benchmark](./figures/fig_quantitative2.png) | [HR Benchmark (Trained with extra DB)](./figures/fig_quantitative3.png) | [DIS](./figures/fig_quantitative4.png)
:-:|:-:|:-:|:-:
<img src=./figures/fig_quantitative.png height=200px> | <img src=./figures/fig_quantitative2.png height=200px> | <img src=./figures/fig_quantitative3.png height=200px> | <img src=./figures/fig_quantitative4.png height=200px>

</p>
</details>

### Qualitative Results

[DAVIS-S & HRSOD](./figures/fig_qualitative.png) | [UHRSD](./figures/fig_qualitative2.png) | [UHRSD (w/ HR scale)](./figures/fig_qualitative3.jpg) | [DIS](./figures/fig_qualitative_dis.png)
:-:|:-:|:-:|:-:
<img src=./figures/fig_qualitative.png height=200px> | <img src=./figures/fig_qualitative2.png height=200px> | <img src=./figures/fig_qualitative3.jpg height=200px> | <img src=./figures/fig_qualitative_dis.png height=200px>

## Citation

```
@inproceedings{kim2022revisiting,
  title={Revisiting Image Pyramid Structure for High Resolution Salient Object Detection},
  author={Kim, Taehun and Kim, Kunhee and Lee, Joonyeong and Cha, Dongmin and Lee, Jiho and Kim, Daijin},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={108--124},
  year={2022}
}
```

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) 
(No.2017-0-00897, Development of Object Detection and Recognition for Intelligent Vehicles) and 
(No.B0101-15-0266, Development of High Performance Visual BigData Discovery Platform for Large-Scale Realtime Data Analysis)

### Special Thanks to :tada:
* [TasksWithCode](https://github.com/taskswithcode) team for sharing our work and making the most amazing web app demo.


## References

### Related Works

* Towards High-Resolution Salient Object Detection ([paper](https://drive.google.com/open?id=15o-Fel0BSyNulGoptrxfHR0t22qMHlTr) | [github](https://github.com/yi94code/HRSOD))
* Disentangled High Quality Salient Object Detection ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_Disentangled_High_Quality_Salient_Object_Detection_ICCV_2021_paper.pdf) | [github](https://github.com/luckybird1994/HQSOD))
* Pyramid Grafting Network for One-Stage High Resolution Saliency Detection ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Pyramid_Grafting_Network_for_One-Stage_High_Resolution_Saliency_Detection_CVPR_2022_paper.pdf) | [github](https://github.com/iCVTEAM/PGNet))

### Resources

* Backbones: [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels), [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

* Datasets
  * LR Benchmarks: [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](http://cbi.gatech.edu/salobj/)
  * HR Benchmarks: [DAVIS-S, HRSOD](https://github.com/yi94code/HRSOD), [UHRSD](https://github.com/iCVTEAM/PGNet)
  * Dichotomous Image Segmentation: [DIS5K](https://xuebinqin.github.io/dis/index.html)

* Evaluation Toolkit
  * SOD Metrics (e.g., S-measure): [PySOD Metrics](https://github.com/lartpang/PySODMetrics)
  * Boundary Metric (mBA): [CascadePSP](https://github.com/hkchengrex/CascadePSP)
