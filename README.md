# Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)

PyTorch implementation of Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)

## Abstract

  Salient object detection (SOD) has been in the spotlight recently, yet has been studied less for high-resolution (HR) images. 
  Unfortunately, HR images and their pixel-level annotations are certainly more labor-intensive and time-consuming compared to low-resolution (LR) images.
  Therefore, we propose an image pyramid-based SOD framework, Inverse Saliency Pyramid Reconstruction Network (InSPyReNet), for HR prediction without any of HR datasets.
  We design InSPyReNet to produce a strict image pyramid structure of saliency map, which enables to ensemble multiple results with pyramid-based image blending.
  For HR prediction, we design a pyramid blending method which synthesizes two different image pyramids from a pair of LR and HR scale from the same image to overcome effective receptive field (ERF) discrepancy. Our extensive evaluation on public LR and HR SOD benchmarks demonstrates that InSPyReNet surpasses the State-of-the-Art (SotA) methods on various SOD metrics and boundary accuracy.

## Architecture

InSPyReNet                 |  pyramid blending
:-------------------------:|:-------------------------:
![](./figures/fig_architecture.png)  |  ![](./figures/fig_pyramid_blending.png)

## 1. Create environment
  + Create conda environment with following command `conda create -y -n inspyrenet python=3.8`
  + Activate environment with following command `conda activate inspyrenet`
  + Install requirements with following command `pip install -r requirements.txt`
  
## 2. Preparation
URL                      |  Destination Folder
:-|:-
[Train Datasets](https://drive.google.com/file/d/1Dxt9pe3uvI3Ow5hEXEzH1q3UwEDYzWjt/view?usp=sharing) | `data/Train_Dataset/...`
[Test Datasets](https://drive.google.com/file/d/1UKJXVnaBgT8ihp3QTOV9NQ1n-UklGbdl/view?usp=sharing) | `data/Test_Dataset/...`
[Res2Net50 checkpoint](https://drive.google.com/file/d/1MMhioAsZ-oYa5FpnTi22XBGh5HkjLX3y/view?usp=sharing) | `data/backbone_ckpt/*.pth`
[SwinB checkpoint](https://drive.google.com/file/d/1fBJFMupe5pV-Vtou-k8LTvHclWs0y1bI/view?usp=sharing) | `data/backbone_ckpt/*.pth`
  
  * Train with extra training datasets (HRSOD, UHRSD):
  ```
  Train:
    Dataset:
        type: "RGB_Dataset"
        root: "data/RGB_Dataset/Train_Dataset"
        sets: ['DUTS-TR'] --> ['DUTS-TR', 'HRSOD-TR-LR', 'UHRSD-TR-LR']
  ```

## 3. Train & Evaluate
  * Train InSPyReNet (SwinB)
  ```
  python run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
  * Inference for test benchmarks
  ```
  python run/Test.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
  * Evaluate metrics
  ```
  python run/Eval.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```

## 4. Checkpoints

### Trained with LR dataset only (DUTS-TE, $384 \times 384$)

Model                      |  Train DB                          
:-|:-
[InSPyReNet (Res2Net50)](https://drive.google.com/file/d/12moRuU8F0-xRvE16bVg6mkGWDuqYHJor/view?usp=sharing) | DUTS-TR                             
[InSPyReNet (SwinB)](https://drive.google.com/file/d/1k5hNJImgEgSmz-ZeJEEb_dVkrOnswVMq/view?usp=sharing) | DUTS-TR

### Trained with LR+HR dataset (with LR scale $384 \times 384$)

Model                      |  Train DB                          
:-|:-
[InSPyReNet (SwinB)](https://drive.google.com/file/d/1nbs6Xa7NMtcikeHFtkQRVrsHbBRHtIqC/view?usp=sharing)         | DUTS-TR, HRSOD-TR-LR                
[InSPyReNet (SwinB)](https://drive.google.com/file/d/1uLSIYXlRsZv4Ho0C-c87xKPhmF_b-Ll4/view?usp=sharing)         | HRSOD-TR-LR, UHRSD-TR-LR            
[InSPyReNet (SwinB)](https://drive.google.com/file/d/14gRNwR7XwJ5oEcR4RWIVbYH3HEV6uBUq/view?usp=sharing)         | DUTS-TR, HRSOD-TR-LR, UHRSD-TR-LR

* LR denotes resized into low-resolution scale (i.e. $384 \times 384$).

### Trained with LR+HR dataset (with HR scale $1024 \times 1024$)

Model                      |  Train DB                          
:-|:-
[InSPyReNet (SwinB)](https://drive.google.com/file/d/1UBGFDUYZ9SysZr96dhsscZg7nDXt6IUD/view?usp=sharing)         | DUTS-TR, HRSOD-TR
[InSPyReNet (SwinB)](https://drive.google.com/file/d/1HB02tiInEgo-pNzwqyvyV6eSN1Y2xPRJ/view?usp=sharing)         | HRSOD-TR, UHRSD-TR

## 5. Results

* Quantitative

![](./figures/fig_quantitative.png) 

* Qualitative
  
![](./figures/fig_qualitative.png) 
## 5. Citation

+ Backbones:
  + Res2Net: [A New Multi-scale Backbone Architecture](https://github.com/Res2Net/Res2Net-PretrainedModels)
  + Swin Transformer: [Hierarchical Vision Transformer using Shifted Windows](https://github.com/microsoft/Swin-Transformer)
+ Datasets:
  + [DUTS](http://saliencydetection.net/duts/)
  + [DUT-OMRON](http://saliencydetection.net/dut-omron/)
  + [ECSSD](https://i.cs.hku.hk/~gbli/deep_saliency.html)
  + [HKU-IS](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
  + [PASCAL-S](http://cbi.gatech.edu/salobj/)
  + [DAVIS-S, HRSOD](https://github.com/yi94code/HRSOD)
  + [UHRSD](https://github.com/iCVTEAM/PGNet)

+ Evaluation Toolkit: [PySOD Metrics](https://github.com/lartpang/PySODMetrics)
