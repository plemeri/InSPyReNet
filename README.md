# Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)

PyTorch implementation of Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)

To appear in the 16th Asian Conference on Computer Vision (ACCV2022)

[[Arxiv](https://arxiv.org/abs/2209.09475)]

> **Abstract:**
  Salient object detection (SOD) has been in the spotlight recently, yet has been studied less for high-resolution (HR) images. 
  Unfortunately, HR images and their pixel-level annotations are certainly more labor-intensive and time-consuming compared to low-resolution (LR) images.
  Therefore, we propose an image pyramid-based SOD framework, Inverse Saliency Pyramid Reconstruction Network (InSPyReNet), for HR prediction without any of HR datasets.
  We design InSPyReNet to produce a strict image pyramid structure of saliency map, which enables to ensemble multiple results with pyramid-based image blending.
  For HR prediction, we design a pyramid blending method which synthesizes two different image pyramids from a pair of LR and HR scale from the same image to overcome effective receptive field (ERF) discrepancy. Our extensive evaluation on public LR and HR SOD benchmarks demonstrates that InSPyReNet surpasses the State-of-the-Art (SotA) methods on various SOD metrics and boundary accuracy.

## Architecture

### InSPyReNet
<div align="center">
  <img src="./figures/fig_architecture.png" width="1200px">
</div>

### pyramid blending

<div align="center">
  <img src="./figures/fig_pyramid_blending.png" width="600px">
</div>

## 1. Create environment
  + Create conda environment with following command `conda create -y -n inspyrenet python=3.8`
  + Activate environment with following command `conda activate inspyrenet`
  + Install requirements with following command `pip install -r requirements.txt`
  
## 2. Preparation
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
Train Datasets | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EVsFkbokdZhGu-Xc5CQaDzQBEn5YRGpTqkBF0qZJYb4PaA?e=FSytKx) | [Link](https://drive.google.com/file/d/1Dxt9pe3uvI3Ow5hEXEzH1q3UwEDYzWjt/view?usp=sharing) 
Test Datasets | `data/Test_Dataset/...`    | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Edc1cQwr5_BItpauYpGksYcBAbaVpLFIVzWoWxrVWIJ8xg?e=Dla9fV) | [Link](https://drive.google.com/file/d/1UKJXVnaBgT8ihp3QTOV9NQ1n-UklGbdl/view?usp=sharing) 
Res2Net50 checkpoint | `data/backbone_ckpt/*.pth` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUO7GDBwoC9CulTPdnq_yhQBlc0SIyyELMy3OmrNhOjcGg?e=T3PVyG) | [Link](https://drive.google.com/file/d/1MMhioAsZ-oYa5FpnTi22XBGh5HkjLX3y/view?usp=sharing)
SwinB checkpoint | `data/backbone_ckpt/*.pth` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESlYCLy0endMhcZm9eC2A4ABatxupp4UPh03EcqFjbtSRw?e=7y6lLt) | [Link](https://drive.google.com/file/d/1fBJFMupe5pV-Vtou-k8LTvHclWs0y1bI/view?usp=sharing)
  
## 3. Train & Evaluate
  * Train InSPyReNet (SwinB)
  ```
  python run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
  * Train with extra training datasets is just changing [this line](https://github.com/plemeri/InSPyReNet/blob/217d7d4944506870a60987533af685140a0bc642/configs/InSPyReNet_SwinB.yaml#L12), simply adding more directories  (HRSOD, UHRSD):
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

   * Please note that we only uploaded the low-resolution (LR) version of HRSOD and UHRSD due to their large image resolution. In order to use them, please download them from the original repositories (see references below), and change the directory names as we did to the LR versions.

## 4. Inference on your own data
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

## 5. Checkpoints

* If you want to try our trained checkpoints below, please make sure to locate `latest.pth` file to the [correct location](https://github.com/plemeri/InSPyReNet/blob/217d7d4944506870a60987533af685140a0bc642/configs/InSPyReNet_SwinB.yaml#L72). 

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
SwinB | DUTS-TR, DUTS-TE, FSS-1000, MSRA-10K, ECSSD, HRSOD-TR-LR, UHRSD-TR-LR | [InSPyReNet_SwinB.yaml](configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESKuh1zhToVFsIxhUUsgkbgBnu2kFXCFLRuSz1xxsKzjhA?e=02HDrm) | [Link](https://drive.google.com/file/d/1iRX-0MVbUjvAVns5MtVdng6CQlGOIo3m/view?usp=sharing) 

## 6. Results

* Quantitative
  * Trained with DUTS-TR only
  ![](./figures/fig_quantitative.png) 
  ![](./figures/fig_quantitative2.png) 
  
  * Trained with extra datasets
  ![](./figures/fig_quantitative3.png) 

* Qualitative

  * Trained with with DUTS-TR only
  ![](./figures/fig_qualitative.png)
  ![](./figures/fig_qualitative2.png)

  * Trained with extra datasets
    
    * Orders for each sample (\* denotes LR scale for training time. Please refer to the Table 3. in the supplementary material)
    
    Image         | GT               | PGNet (H, U) | Ours (D, H*) 
    ------ | -----  | -----  | ----- 
    Ours (H*, U*) | Ours (D, H*, U*) | Ours (D, H)  | Ours (H, U) 
      
    ![](./figures/fig_qualitative3.jpg)

## 7.Citation

```
```

## References

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
