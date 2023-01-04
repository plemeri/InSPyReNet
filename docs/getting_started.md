# Getting Started :flight_departure:

## Create environment
  * Create conda environment with following command `conda create -y -n inspyrenet python`
  * Activate environment with following command `conda activate inspyrenet`
  * Install latest PyTorch from [Official Website](https://pytorch.org/get-started/locally/)
    * Linux [2022.11.09]
    ```
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    ```
  * Install requirements with following command `pip install -r requirements.txt`
  
## Preparation

* For training, you may need training datasets and ImageNet pre-trained checkpoints for the backbone. For testing (inference), you may need test datasets (sample images).
* Training datasets are expected to be located under [Train.Dataset.root](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L11). Likewise, testing datasets should be under [Test.Dataset.root](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L55).
* Each dataset folder should contain `images` folder and `masks` folder for images and ground truth masks respectively.
* You may use multiple training datasets by listing dataset folders for [Train.Dataset.sets](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L12), such as `[DUTS-TR] -> [DUTS-TR, HRSOD-TR, UHRSD-TR]`.

### Backbone Checkpoints
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
Res2Net50 checkpoint | `data/backbone_ckpt/*.pth` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUO7GDBwoC9CulTPdnq_yhQBlc0SIyyELMy3OmrNhOjcGg?e=T3PVyG&download=1) | [Link](https://drive.google.com/file/d/1MMhioAsZ-oYa5FpnTi22XBGh5HkjLX3y/view?usp=sharing)
SwinB checkpoint     | `data/backbone_ckpt/*.pth` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESlYCLy0endMhcZm9eC2A4ABatxupp4UPh03EcqFjbtSRw?e=7y6lLt&download=1) | [Link](https://drive.google.com/file/d/1fBJFMupe5pV-Vtou-k8LTvHclWs0y1bI/view?usp=sharing)

* We changed Res2Net50 checkpoint to resolve an error while training with DDP. Please refer to [issue #9](https://github.com/plemeri/InSPyReNet/issues/9).

### Train Datasets
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
DUTS-TR | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQ7L2XS-5YFMuJGee7o7HQ8BdRSLO8utbC_zRrv-KtqQ3Q?e=bCSIeo&download=1) | [Link](https://drive.google.com/file/d/1hy5UTq65uQWFO5yzhEn9KFIbdvhduThP/view?usp=share_link)

### Extra Train Datasets (High Resolution, Optional)
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
HRSOD-TR | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUx92hUgZJNrWPj46PC0yEBXcorQskXOCSz8SnGH5AcLQ?e=WA5pc6&download=1) | N/A
UHRSD-TR | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea4_UCbsKmhKnMCccAJOTLgBmQFsQ4KhJSf2jx8WQqj3Wg?e=18kYZS&download=1) | N/A
DIS-TR   | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZtZJ493tVNJjBIpNLdus68B3u906PdWtHsf87pulj78jw?e=bUg2UQ&download=1) | N/A

### Test Datasets
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
DUTS-TE   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfuCxjveXphPpIska9BxHDMBHpYroEKdVlq9HsonZ4wLDw?e=Mz5giA&download=1) | [Link](https://drive.google.com/file/d/1w4pigcQe9zplMulp1rAwmp6yYXmEbmvy/view?usp=share_link) 
DUT-OMRON | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERvApm9rHH5LiR4NJoWHqDoBCneUQNextk8EjQ_Hy0bUHg?e=wTRZQb&download=1) | [Link](https://drive.google.com/file/d/1qIm_GQLLQkP6s-xDZhmp_FEAalavJDXf/view?usp=sharing) 
ECSSD     | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ES_GCdS0yblBmnRaDZ8xmKQBPU_qeECTVB9vlPUups8bnA?e=POVAlG&download=1) | [Link](https://drive.google.com/file/d/1qk_12KLGX6FPr1P_S9dQ7vXKaMqyIRoA/view?usp=sharing) 
HKU-IS    | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYBRVvC1MJRAgSfzt0zaG94BU_UWaVrvpv4tjogu4vSV6w?e=TKN7hQ&download=1) | [Link](https://drive.google.com/file/d/1H3szJYbr5_CRCzrYfhPHThTgszkKd1EU/view?usp=share_link) 
PASCAL-S  | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUDGDckMnZHhEPy8YQGwBQB5MN3qInBkEygpIr7ccJdTQ?e=YarZaQ&download=1) | [Link](https://drive.google.com/file/d/1h0IE2DlUt0HHZcvzMV5FCxtZqQqh9Ztf/view?usp=sharing)
DAVIS-S   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ebam8I2o-tRJgADcq-r9YOkBCDyaAdWBVWyfN-xCYyAfDQ?e=Mqz8cK&download=1) | [Link](https://drive.google.com/file/d/15F0dy9o02LPTlpUbnD9NJlGeKyKU3zOz/view?usp=sharing)
HRSOD-TE  | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EbHOQZKC59xIpIdrM11ulWsBHRYY1wZY2njjWCDFXvT6IA?e=wls17m&download=1) | [Link](https://drive.google.com/file/d/1KnUCsvluS4kP2HwUFVRbKU8RK_v6rv2N/view?usp=sharing)
UHRSD-TE  | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUpc8QJffNpNpESv-vpBi40BppucqOoXm_IaK7HYJkuOog?e=JTjGmS&download=1) | [Link](https://drive.google.com/file/d/1niiHBo9LX6-I3KsEWYOi_s6cul80IYvK/view?usp=sharing)

### Extra Test Datasets (Optional) 
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
FSS-1000  | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EaP6DogjMAVCtTKcC_Bx-YoBoBSWBo90lesVcMyuCN35NA?e=0DDohA&download=1) | N/A
MSRA-10K  | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EauXsIkxBzhDjio6fW0TubUB4L7YJc0GMTaq7VfjI2nPsg?e=c5DIxg&download=1) | N/A
DIS-VD    | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYJm3BqheaxNhdVoMt6X41gBgVnE4dulBwkp6pbOQtcIrQ?e=T6dtXm&download=1) | [Link](https://drive.google.com/file/d/1jhlZb3QyNPkc0o8nL3RWF0MLuVsVtJju/view?usp=sharing)
DIS-TE1   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcGYE_Gc0cVHoHi_qUtmsawB_5v9RSpJS5PIAPlLu6xo9A?e=Nu5mkQ&download=1) | [Link](https://drive.google.com/file/d/1iz8Y4uaX3ZBy42N2MIOkmNb0D5jroFPJ/view?usp=sharing)
DIS-TE2   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdhgMdbZ049GvMv7tNrjbbQB1wL9Ok85YshiXIkgLyTfkQ?e=mPA6Po&download=1) | [Link](https://drive.google.com/file/d/1DWSoWogTWDuS2PFbD1Qx9P8_SnSv2zTe/view?usp=sharing)
DIS-TE3   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcxXYC_3rXxKsQBrp6BdNb4BOKxBK3_vsR9RL76n7YVG-g?e=2M0cse&download=1) | [Link](https://drive.google.com/file/d/1bIVSjsxCjMrcmV1fsGplkKl9ORiiiJTZ/view?usp=sharing)
DIS-TE4   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdkG2SUi8flJvoYbHHOmvMABsGhkCJCsLLZlaV2E_SZimA?e=zlM2kC&download=1) | [Link](https://drive.google.com/file/d/1VuPNqkGTP1H4BFEHe807dTIkv8Kfzk5_/view?usp=sharing)

## Train & Evaluate

  * Train InSPyReNet
  ```
  # Single GPU
  python run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose
  
  # Multi GPUs with DDP (e.g., 4 GPUs)
  torchrun --standalone --nproc_per_node=4 run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose

  # Multi GPUs with DDP with designated devices (e.g., 2 GPUs - 0 and 1)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```

  * `--config [CONFIG_FILE], -c [CONFIG_FILE]`: config file path for training.
  * `--resume, -r`: use this argument to resume from last checkpoint.
  * `--verbose, -v`: use this argument to output progress info.
  * `--debug, -d`: use this argument to save debug images every epoch.

  * Train with extra training datasets can be done by just changing [Train.Dataset.sets](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L12) in the `yaml` config file, which is just simply adding more directories (e.g., HRSOD-TR, HRSOD-TR, UHRSD-TR, ...):
   ```
   Train:
     Dataset:
         type: "RGB_Dataset"
         root: "data/RGB_Dataset/Train_Dataset"
         sets: ['DUTS-TR'] --> ['DUTS-TR', 'HRSOD-TR', 'UHRSD-TR']
   ```
  * Inference for test benchmarks
  ```
  python run/Test.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
  * Evaluate metrics
  ```
  python run/Eval.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```

  * All-in-One command (Train, Test, Eval in single command)
  ```
  # Single GPU
  python Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose

  # Multi GPUs with DDP (e.g., 4 GPUs)
  torchrun --standalone --nproc_per_node=4 Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose

  # Multi GPUs with DDP with designated devices (e.g., 2 GPUs - 0 and 1)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
## Inference on your own data
  * You can inference your own single image or images (.jpg, .jpeg, and .png are supported), single video or videos (.mp4, .mov, and .avi are supported), and webcam input (ubuntu and macos are tested so far).
  ```
  python run/Inference.py --config configs/InSPyReNet_SwinB.yaml --source [SOURCE] --dest [DEST] --type [TYPE] --gpu --jit --verbose
  ```

  * `--source [SOURCE]`: Specify your data in this argument.
    * Single image - `image.png`
    * Folder containing images - `path/to/img/folder`
    * Single video - `video.mp4`
    * Folder containing videos - `path/to/vid/folder`
    * Webcam input: `0` (may vary depends on your device.)
  * `--dest [DEST]` (optional): Specify your destination folder. If not specified, it will be saved in `results` folder.
  * `--type [TYPE]`: Choose between `map` `green`, `rgba`, `blur`, `overlay`, and another image file.
    * `map` will output saliency map only. 
    * `green` will change the background with green screen. 
    * `rgba` will generate RGBA output regarding saliency score as an alpha map. Note that this will not work for video and webcam input. 
    * `blur` will blur the background.
    * `overlay` will cover the salient object with translucent green color, and highlight the edges.
    * Another image file (e.g., `backgroud.png`) will be used as a background, and the object will be overlapped on it.
    <details><summary>Examples of TYPE argument</summary>
    <p>
    <img src=../figures/demo_type.png >
    </p>
    </details>
  * `--gpu`: Use this argument if you want to use GPU. 
  * `--jit`: Slightly improves inference speed when used. 
  * `--verbose`: Use when you want to visualize progress.

