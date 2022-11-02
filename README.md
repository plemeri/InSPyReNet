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
<a href="https://arxiv.org/abs/2209.09475"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
<a href=""><img  src="https://img.shields.io/badge/license-MIT-blue"></a>
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

[2022.10.20] :new: We trained our model on [Dichotomous Image Segmentation dataset (DIS5K)](https://xuebinqin.github.io/dis/index.html) and showed competitive results! Trained checkpoint and pre-computed segmentation masks are available in [Checkpoints](#checkpoints) and [Pre-Computed Saliency Maps](#pre-computed-saliency-maps) section. Also, you can check our qualitative and quantitative results in [Results](#results) section.

[2022.10.28] Multi GPU training for latest pytorch is now available.

[2022.10.31] :new: [TasksWithCode](https://github.com/taskswithcode) provided an amazing web demo with [HuggingFace](https://huggingface.co). Visit the [WepApp](https://huggingface.co/spaces/taskswithcode/salient-object-detection) and try with your image! 

## Demo :rocket:

* <img src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg height="20px" width="20px"> Try [WepApp](https://huggingface.co/spaces/taskswithcode/salient-object-detection) on HuggingFace to generate your own results!

[Image Sample](./figures/demo_image.gif) | [Video Sample](./figures/demo_video.gif)
:-:|:-:
<img src=./figures/demo_image.gif height="350px" width="350px"> | <img src=./figures/demo_video.gif height="350px" width="350px">

## Architecture

[InSPyReNet](./figures/fig_architecture.png) | [pyramid blending](./figures/fig_pyramid_blending.png)
:-:|:-:
<img src=./figures/fig_architecture.png height="350px" width="350px"> | <img src=./figures/fig_pyramid_blending.png height="350px" width="350px">

## Create environment
  + Create conda environment with following command `conda create -y -n inspyrenet python=3.8`
  + Activate environment with following command `conda activate inspyrenet`
  + Install requirements with following command `pip install -r requirements.txt`
  
## Preparation

* For training, you may need training datasets and ImageNet pre-trained checkpoints for the backbone. For testing (inference), you may need test datasets (sample images).
* Training datasets are expected to be located under [Train.Dataset.root](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L10). Likewise, testing datasets should be under [Test.Dataset.root](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L58).
* Each dataset folder should contain `images` folder and `masks` folder for images and ground truth masks respectively.
* You may use multiple training datasets by listing dataset folders for [Train.Dataset.sets](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L12), such as `[DUTS-TR] -> [DUTS-TR, HRSOD-TR, UHRSD-TR]`.


Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
Train Datasets | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EVsFkbokdZhGu-Xc5CQaDzQBEn5YRGpTqkBF0qZJYb4PaA?e=FSytKx) | [Link](https://drive.google.com/file/d/1Dxt9pe3uvI3Ow5hEXEzH1q3UwEDYzWjt/view?usp=sharing) 
Test Datasets | `data/Test_Dataset/...`    | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Edc1cQwr5_BItpauYpGksYcBAbaVpLFIVzWoWxrVWIJ8xg?e=Dla9fV) | [Link](https://drive.google.com/file/d/1UKJXVnaBgT8ihp3QTOV9NQ1n-UklGbdl/view?usp=sharing) 
Res2Net50 checkpoint | `data/backbone_ckpt/*.pth` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUO7GDBwoC9CulTPdnq_yhQBlc0SIyyELMy3OmrNhOjcGg?e=T3PVyG) | [Link](https://drive.google.com/file/d/1MMhioAsZ-oYa5FpnTi22XBGh5HkjLX3y/view?usp=sharing)
SwinB checkpoint | `data/backbone_ckpt/*.pth` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESlYCLy0endMhcZm9eC2A4ABatxupp4UPh03EcqFjbtSRw?e=7y6lLt) | [Link](https://drive.google.com/file/d/1fBJFMupe5pV-Vtou-k8LTvHclWs0y1bI/view?usp=sharing)
  
## Train & Evaluate

  * Train InSPyReNet
  ```
  # Single GPU
  python run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose
  
  # Multi GPUs with DDP 
  torchrun --standalone --nproc_per_node=[NUM_GPU] Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```

  * Train with extra training datasets can be done by just changing [Train.Dataset.sets](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L12) in the `yaml` config file, which is just simply adding more directories (e.g., HRSOD-TR, HRSOD-TR-LR, UHRSD-TR, ...):
   ```
   Train:
     Dataset:
         type: "RGB_Dataset"
         root: "data/RGB_Dataset/Train_Dataset"
         sets: ['DUTS-TR'] --> ['DUTS-TR', 'HRSOD-TR-LR', 'UHRSD-TR-LR']
   ```
  * Inference for test benchmarks
  ```
  python run/Test.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
  * Evaluate metrics
  ```
  python run/Eval.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```

  * All-in-One command (Train, Test, Eval at the same time)
  ```
  # Single GPU
  python Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose

  # Multi GPUs with DDP 
  torchrun --standalone --nproc_per_node=[NUM_GPU] Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```


   * Please note that we only uploaded the low-resolution (LR) version of HRSOD and UHRSD due to their large image resolution. In order to use them, please download them from the original repositories (see references below), and change the directory names as we did to the LR versions.

## Inference on your own data
  + You can inference your own single image or images (.jpg, .jpeg, and .png are supported), single video or videos (.mp4, .mov, and .avi are supported), and webcam input (ubuntu and macos are tested so far).
  + `python run/Inference.py --config configs/InSPyReNet_SwinB.yaml --source [SOURCE] --dest [DEST] --type [TYPE] --gpu --jit --verbose`
    + SOURCE: Specify your data in this argument.
      + Single image - `image.png`
      + Folder containing images - `path/to/img/folder`
      + Single video - `video.mp4`
      + Folder containing videos - `path/to/vid/folder`
      + Webcam input: `0` (may vary depends on your device.)
    + DEST (optional): Specify your destination folder. If not specified, it will be saved in `results` folder.
    + TYPE: Choose between `map, green, rgba, blur`
      + `map` will output saliency map only. 
      + `green` will change the background with green screen. 
      + `rgba` will generate RGBA output regarding saliency score as an alpha map. Note that this will not work for video and webcam input. 
      + `blur` will blur the background.
    + --gpu: Use this argument if you want to use GPU. 
    + --jit: Slightly improves inference speed when used. 
    + --verbose: Use when you want to visualize progress.

## Checkpoints

Note: If you want to try our trained checkpoints below, please make sure to locate `latest.pth` file to the [Test.Checkpoint.checkpoint_dir](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L72). 

### Trained with LR dataset only (DUTS-TR, 384 X 384)

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
Res2Net50 | DUTS-TR | [InSPyReNet_Res2Net50](configs/InSPyReNet_Res2Net50.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERqm7RPeNBFPvVxkA5P5G2AB-mtFsiYkCNHnBf0DcwpFzw?e=nayVno) | [Link](https://drive.google.com/file/d/12moRuU8F0-xRvE16bVg6mkGWDuqYHJor/view?usp=sharing)
SwinB | DUTS-TR | [InSPyReNet_SwinB.yaml](configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EV0ow4E8LddCgu5tAuAkMbcBpBYoEDmJgQg5wkiuvLoQUA?e=cOZspv) | [Link](https://drive.google.com/file/d/1k5hNJImgEgSmz-ZeJEEb_dVkrOnswVMq/view?usp=sharing)

### Trained with LR+HR dataset (with LR scale 384 X 384)

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
SwinB | DUTS-TR, HRSOD-TR-LR | [InSPyReNet_SwinB.yaml](configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EWxPZoIKALlGsfrNgUFNvxwBC8IE8jzzhPNtzcbHmTNFcg?e=e22wmy) | [Link](https://drive.google.com/file/d/1nbs6Xa7NMtcikeHFtkQRVrsHbBRHtIqC/view?usp=sharing) 
SwinB | HRSOD-TR-LR, UHRSD-TR-LR | [InSPyReNet_SwinB.yaml](configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQe-iy0AZctIkgl3o-BmVYUBn795wvii3tsnBq1fNUbc9g?e=gMZ4PV) | [Link](https://drive.google.com/file/d/1uLSIYXlRsZv4Ho0C-c87xKPhmF_b-Ll4/view?usp=sharing) 
SwinB | DUTS-TR, HRSOD-TR-LR, UHRSD-TR-LR | [InSPyReNet_SwinB.yaml](configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfsCbnfAU1RAqCJIkj1ewRgBhFetStsGB6SMSq_UJZimjA?e=Ghuacy) | [Link](https://drive.google.com/file/d/14gRNwR7XwJ5oEcR4RWIVbYH3HEV6uBUq/view?usp=sharing) 

* *-LR denotes resized into low-resolution scale (i.e., 384 X 384).

### Trained with LR+HR dataset (with HR scale 1024 X 1024)

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
SwinB | DUTS-TR, HRSOD-TR | [InSPyReNet_SwinB.yaml](configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EW2Qg-tMBBxNkygMj-8QgMUBiqHox5ExTOJl0LGLsn6AtA?e=Mam8Ur) | [Link](https://drive.google.com/file/d/1UBGFDUYZ9SysZr96dhsscZg7nDXt6IUD/view?usp=sharing) 
SwinB | HRSOD-TR, UHRSD-TR | [InSPyReNet_SwinB.yaml](configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EeE8nnCt_AdFvxxu0JsxwDgBCtGchuUka6DW9za_epX-Qw?e=U7wZu9) | [Link](https://drive.google.com/file/d/1HB02tiInEgo-pNzwqyvyV6eSN1Y2xPRJ/view?usp=sharing)

### Trained with Massive SOD Datasets (with LR scale 384 x 384, Not in the paper, just for fun!)

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
SwinB | DUTS-TR, DUTS-TE, DUT-OMRON, FSS-1000, MSRA-10K, ECSSD, HKU-IS, PASCAL-S, HRSOD-TR-LR, UHRSD-TR-LR | [InSPyReNet_SwinB.yaml](configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESKuh1zhToVFsIxhUUsgkbgBnu2kFXCFLRuSz1xxsKzjhA?e=02HDrm) | [Link](https://drive.google.com/file/d/1iRX-0MVbUjvAVns5MtVdng6CQlGOIo3m/view?usp=sharing)

### :new: Trained with Dichotomous Image Segmentation dataset (DIS5K-TR) with LR scale (384 X 384) [Added in 2022.10.20] 
* If you want to train / inference with DIS5K, you may need to change the subdirectories' names (`im` and `gt`) to our way (`images` and `masks`) for training and testing datasets. Please refer to the [Preparation](#preparation) section.

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
SwinB | DIS5K-TR | [InSPyReNet_SwinB_DIS5K.yaml](configs/InSPyReNet_SwinB_DIS5K.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERKrQ_YeoJRHl_3HcH8ZJLoBedsa6hZlmIIf66wobZRGuw?e=EywJmS) | [Link](https://drive.google.com/file/d/1Sj7GZoocGMHyKNhFnQQc1FTs76ysJIX3/view?usp=sharing)


## Pre-Computed Saliency Maps

Note: Due to the cloud memory shortage, we only provide results trained on DUTS-TR only. Please generate yourself for the models with extra training datasets if you need. 

Backbone | DUTS-TE | DUT-OMRON | ECSSD | HKU-IS | PASCAL-S | DAVIS-S | HRSOD-TE | UHRSD-TE
:-|:-|:-|:-|:-|:-|:-|:-|:-
Res2Net50 | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Eb0iKXGX1vxEjPhe9KGBKr0Bv7v2vv6Ua5NFybwc6aIi1w?e=oHnGyJ) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ef1HaYMvgh1EuuOL8bw3JGYB41-yo6KdTD8FGXcFZX3-Bg?e=TkW2m8) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdEQQ8o-yI9BtTpROcuB_iIBFSIk0uBJAkNyob0WI04-kw?e=cwEj2V) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ec6LyrumVZ9PoB2Af0OW4dcBrDht0OznnwOBYiu8pdyJ4A?e=Y04Fmn) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETPijMHlTRZIjqO5H4LBknUBmy8TGDwOyUQ1H4EnIpHVOw?e=k1afrh) | N/A | N/A | N/A |
SwinB | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETumLjuBantLim4kRqj4e_MBpK_X5XrTwjGQUToN8TKVjw?e=ZT8AWy) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZbwxhwT6dtHkBJrIMMjTnkBK_HaDTXgHcDSjxuswZKTZw?e=9XeE4b) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESfQK-557uZOmUwG5W49j0EBK42_7dMOaQcPsc_U1zsYlA?e=IvjkKX) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EURH96JUp55EgUHI0A8RzKoBBqvQc1nVb_a67RgwOY7f-w?e=IP9xKa) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EakMpwONph9EmnCM2rS3hn4B_TL42T6tuLjBEeEa5ndkIw?e=XksfA5) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETUCKFX0k8lAvpsDj5sT23QB2ohuE_ST7oQnWdaW7AoCIw?e=MbSmM2) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea6kf6Kk8fpIs15WWDfJMoYBeQUeo9WXvYx9oM5yWFE1Jg?e=RNN0Ns) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EVJLvAP3HwtHksZMUolIfCABHqP7GgAWcG_1V5T_Xrnr2g?e=ct3pzo) |

### * :new: DIS5K Results [Added in 2022.10.20]

Backbone | DIS-VD | DIS-TE1 | DIS-TE2 | DIS-TE3 | DIS-TE4
:-|:-|:-|:-|:-|:-|
SwinB | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUbzddb_QRRCtnXC8Xl6vZoBC6IqOfom52BWbzOYk-b2Ow?e=aqJYi1) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESeW_SOD26tHjBLymmgFaXwBIJlljzNycaGWXLpOp_d_kA?e=2EyMai) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYWT5fZDjI5Bn-lr-iQM1TsB1num0-UqfJC1TIv-LuOXoA?e=jCcnty) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQXm1DEBfaNJmH0B-A3o23kBn4v5j53kP2nF9CpG9SQkyw?e=lEUiZh) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZeH2ufGsFZIoUh6D8Rtv88BBF_ddQXav4xYXXRP_ayEAg?e=AMzIp8)

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

+ Backbones: [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels), [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
+ Datasets:
  + LR Benchmarks: [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [ECSSD](https://i.cs.hku.hk/~gbli/deep_saliency.html), [HKU-IS](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [PASCAL-S](http://cbi.gatech.edu/salobj/)
  + HR Benchmarks: [DAVIS-S, HRSOD](https://github.com/yi94code/HRSOD), [UHRSD](https://github.com/iCVTEAM/PGNet)
  + Dichotomous Image Segmentation: [DIS5K](https://xuebinqin.github.io/dis/index.html)

+ Evaluation Toolkit
  + SOD Metrics (e.g., S-measure): [PySOD Metrics](https://github.com/lartpang/PySODMetrics)
  + Boundary Metric (mBA): [CascadePSP](https://github.com/hkchengrex/CascadePSP)
