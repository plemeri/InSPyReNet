# InSPyReNet: Inverse Saliency Pyramid Reconstruction Network for Salient Object Detection
<p align="center">

  <img src="./figures/figure1.png" alt="Logo" width="300" height="auto">

</p>
PyTorch implementation of InSPyReNet: Inverse Saliency Pyramid Reconstruction Network for Salient Object Detection



## Abstract

Salient object detection (SOD) requires multi-scale features from intermediate backbone feature maps and carefully designed feature aggregation modules to obtain detailed saliency prediction. High-level feature maps from the backbone have high-level semantic information while lack of high frequency details like sharp edges or contours of salient objects. On the other hand, low-level feature maps from the backbone have high frequency details while lack of semantic information for accurate saliency score prediction. Rather than combining feature maps of multiple scales, we separately use them to predict the saliency map at their maximum capabilities of spatial dimension and high frequency details. Inspired by image pyramid, we propose Inverse Saliency Pyramid Reconstruction Network (InSPyReNet), a coarse to fine SOD network with Laplacian pyramid on each scale of saliency map to reconstruct a saliency map to the input image size. While image pyramids can only be constructed from the largest image to smaller scales, our method explicitly estimates Laplacian saliency map from the smallest scale and reconstructs to the larger scales. Our extensive evaluation on five public SOD benchmarks demonstrates our method surpasses the state-of-the-art performance on various SOD metrics.

## Architecture

![Teaser](./figures/figure2.png)

## 1. Create environment
  + Create conda environment with following command `conda create -y -n inspyrenet python=3.8`
  + Activate environment with following command `conda activate inspyrenet`
  + Install requirements with following command `pip install -r requirements.txt`
  
## 2. Preparation
  + Use following command to automatically download datasets and checkpoints with following command `sh install.sh`
  + Instead, you can download them manually.
    + Download datasets and backbone checkpoints from following [URL](https://drive.google.com/file/d/1KkXffb1DEu1be7NO-RPUy1r2bZqJRuYl/view?usp=sharing)
    + Move folder `data` to the repository.
    + Folder should be ordered as follows,
  ```
  .
  ├── configs
  │   ├── InSPyReNet_Res2Net50.yaml
  │   └── InSPyReNet_SwinB.yaml
  ├── data
  │   ├── backbone_ckpt
  │   │   ├── swin_base_patch4_window12_384_22kto1k.pth
  │   │   └── res2net50_v1b_26w_4s-3cf99910.pth
  │   ├── RGB_Dataset
  │   │   ├── Test_Dataset
  │   │   │   ├── DUTS-TE
  │   │   │   ├── DUT-OMRON
  │   │   │   ├── ECSSD
  │   │   │   ├── HKU-IS
  │   │   │   └──PASCAL-S
  │   │   └── Train_Dataset
  │   │       └── DUTS-TR
  ├── Expr.py
  ├── figures
  │   ├── figure1.png
  │   ├── figure2.png
  │   └── results.png
  ├── lib
  │   ├── backbones
  │   ├── __init__.py
  │   ├── InSPyReNet_Res2Net50.py
  │   ├── InSPyReNet_SwinB.py
  │   ├── losses
  │   ├── modules
  ├── LICENSE
  ├── README.md
  ├── requirements.txt
  ├── results
  ├── run
  │   ├── Eval.py
  │   ├── __init__.py
  │   ├── Test.py
  │   └── Train.py
  ├── snapshots
  │   ├── InSPyReNet_Res2Net50
  │   │   ├── DUTS-TE
  │   │   ├── DUT-OMRON
  │   │   ├── ECSSD
  │   │   ├── HKU-IS
  │   │   ├── PASCAL-S
  │   │   └── latest.pth
  │   ├── InSPyReNet_SwinB
  │   │   ├── DUTS-TE
  │   │   ├── DUT-OMRON
  │   │   ├── ECSSD
  │   │   ├── HKU-IS
  │   │   ├── PASCAL-S
  │   │   └── latest.pth
  └── utils
      ├── custom_transforms.py
      ├── dataloader.py
      ├── eval_functions.py
      └── utils.py
  ```

## 3. Train & Evaluate
  + You can train with `python run/Train.py --config configs/InSPyReNet_SwinB.yaml`
  + You can generate prediction for test dataset with `python run/Test.py --config configs/InSPyReNet_SwinB.yaml`
  + You can evaluate generated prediction with `python run/Eval.py --config configs/InSPyReNet_SwinB.yaml`
  + You can also use `python Expr.py --config configs/InSPyReNet_SwinB.yaml` to train, generate prediction and evaluation in single command
  
  + (optional) Download our best result checkpoints and pre-computed saliency maps from following [URL](https://drive.google.com/file/d/1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0/view?usp=sharing) for InSPyReNet_Res2Net50 and InSPyReNet_SwinB. Locate pth files following above file location. If you use `install.sh`, then you don't need to download them manually.
  + (optional) You can download pre-computed saliency maps from other methods and evaluate with our evaluation code. Create an yaml file as follows,
  ```
  Eval:
    gt_root: "data/RGB_Dataset/Test_Dataset"
    pred_root: "[DIR_FOR_YOUR_PRE_COMPUTED_SALIENCY_MAP]"
    result_path: "results"
    datasets: ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']
    metrics: ['Sm', 'mae', 'adpEm', 'maxEm', 'avgEm', 'adpFm', 'maxFm', 'avgFm', 'wFm']
  ``` 

## 4. Experimental Results
  + InSPyReNet_Res2Net50
  ```
dataset       Sm    mae     Em    maxF    avgF    wFm    IoUmaxF    maxIoU    meanIoU
---------  -----  -----  -----  ------  ------  -----  ---------  --------  ---------
DUTS-TE    0.901  0.036  0.933   0.902   0.871  0.841      0.800     0.817      0.786
DUT-OMRON  0.845  0.058  0.869   0.819   0.792  0.747      0.702     0.723      0.700
ECSSD      0.938  0.030  0.965   0.957   0.935  0.920      0.893     0.903      0.878
HKU-IS     0.927  0.029  0.962   0.945   0.921  0.902      0.861     0.876      0.849
PASCAL-S   0.872  0.058  0.907   0.888   0.864  0.827      0.773     0.792      0.775
  ```
  + InSPyReNet_SwinB
  ```
dataset       Sm    mae     Em    maxF    avgF    wFm    IoUmaxF    maxIoU    meanIoU
---------  -----  -----  -----  ------  ------  -----  ---------  --------  ---------
DUTS-TE    0.931  0.024  0.964   0.934   0.903  0.889      0.852     0.869      0.836
DUT-OMRON  0.878  0.044  0.907   0.855   0.831  0.801      0.759     0.778      0.754
ECSSD      0.949  0.023  0.974   0.967   0.946  0.937      0.913     0.923      0.900
HKU-IS     0.944  0.022  0.976   0.958   0.932  0.924      0.890     0.904      0.877
PASCAL-S   0.894  0.047  0.933   0.912   0.886  0.859      0.809     0.827      0.810
  ```
  + Qualitative Results 

![results](./figures/results.png)
  
## 5. Citation

+ Res2Net: A New Multi-scale Backbone Architecture [github](https://github.com/Res2Net/Res2Net-PretrainedModels)
+ Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [github](https://github.com/microsoft/Swin-Transformer)
+ Datasets - [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [ECSSD](https://i.cs.hku.hk/~gbli/deep_saliency.html), [HKU-IS](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [PASCAL-S](http://cbi.gatech.edu/salobj/)
+ Evaluation Toolkit: [PySOD Metrics](https://github.com/lartpang/PySODMetrics)