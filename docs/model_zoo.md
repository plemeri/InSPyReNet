# :giraffe: Model Zoo

## :label: Note: If you want to try our trained checkpoints below, please make sure to locate `latest.pth` file to the [Test.Checkpoint.checkpoint_dir](https://github.com/plemeri/InSPyReNet/blob/main/../configs/InSPyReNet_SwinB.yaml#L72). 

### Trained with LR dataset only (DUTS-TR, 384 X 384)

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
Res2Net50 | DUTS-TR | [InSPyReNet_Res2Net50.yaml](../configs/InSPyReNet_Res2Net50.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERqm7RPeNBFPvVxkA5P5G2AB-mtFsiYkCNHnBf0DcwpFzw?e=nayVno) | [Link](https://drive.google.com/file/d/12moRuU8F0-xRvE16bVg6mkGWDuqYHJor/view?usp=sharing)
SwinB | DUTS-TR | [InSPyReNet_SwinB.yaml](../configs/InSPyReNet_SwinB.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EV0ow4E8LddCgu5tAuAkMbcBpBYoEDmJgQg5wkiuvLoQUA?e=cOZspv) | [Link](https://drive.google.com/file/d/1k5hNJImgEgSmz-ZeJEEb_dVkrOnswVMq/view?usp=sharing)

### Trained with LR+HR dataset (with LR scale 384 X 384)

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
SwinB | DUTS-TR, HRSOD-TR | [InSPyReNet_SwinB_DH_LR.yaml](../configs/extra_dataset/InSPyReNet_SwinB_DH_LR.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EWxPZoIKALlGsfrNgUFNvxwBC8IE8jzzhPNtzcbHmTNFcg?e=e22wmy) | [Link](https://drive.google.com/file/d/1nbs6Xa7NMtcikeHFtkQRVrsHbBRHtIqC/view?usp=sharing) 
SwinB | HRSOD-TR, UHRSD-TR | [InSPyReNet_SwinB_HU_LR.yaml](../configs/extra_dataset/InSPyReNet_SwinB_HU_LR.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQe-iy0AZctIkgl3o-BmVYUBn795wvii3tsnBq1fNUbc9g?e=gMZ4PV) | [Link](https://drive.google.com/file/d/1uLSIYXlRsZv4Ho0C-c87xKPhmF_b-Ll4/view?usp=sharing) 
SwinB | DUTS-TR, HRSOD-TR, UHRSD-TR | [InSPyReNet_SwinB_DHU_LR.yaml](../configs/extra_dataset/InSPyReNet_SwinB_DHU_LR.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfsCbnfAU1RAqCJIkj1ewRgBhFetStsGB6SMSq_UJZimjA?e=Ghuacy) | [Link](https://drive.google.com/file/d/14gRNwR7XwJ5oEcR4RWIVbYH3HEV6uBUq/view?usp=sharing) 

### Trained with LR+HR dataset (with HR scale 1024 X 1024)

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
SwinB | DUTS-TR, HRSOD-TR | [InSPyReNet_SwinB_DH.yaml](../configs/extra_dataset/InSPyReNet_SwinB_DH.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EW2Qg-tMBBxNkygMj-8QgMUBiqHox5ExTOJl0LGLsn6AtA?e=Mam8Ur) | [Link](https://drive.google.com/file/d/1UBGFDUYZ9SysZr96dhsscZg7nDXt6IUD/view?usp=sharing) 
SwinB | HRSOD-TR, UHRSD-TR | [InSPyReNet_SwinB_HU.yaml](../configs/extra_dataset/InSPyReNet_SwinB_HU.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EeE8nnCt_AdFvxxu0JsxwDgBCtGchuUka6DW9za_epX-Qw?e=U7wZu9) | [Link](https://drive.google.com/file/d/1HB02tiInEgo-pNzwqyvyV6eSN1Y2xPRJ/view?usp=sharing)

### <img src="https://www.kindpng.com/picc/b/124-1249525_all-might-png.png" width=20px> Trained with Massive SOD Datasets (with LR scale 384 x 384, Not in the paper, just for fun!)

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
SwinB | DUTS-TR, DUTS-TE, DUT-OMRON, FSS-1000, MSRA-10K, ECSSD, HKU-IS, PASCAL-S, HRSOD-TR, UHRSD-TR | [Plus_Ultra.yaml](../configs/extra_dataset/Plus_Ultra.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESKuh1zhToVFsIxhUUsgkbgBnu2kFXCFLRuSz1xxsKzjhA?e=02HDrm) | [Link](https://drive.google.com/file/d/1iRX-0MVbUjvAVns5MtVdng6CQlGOIo3m/view?usp=sharing)

### :new: Trained with Dichotomous Image Segmentation dataset (DIS5K-TR) with LR scale (384 X 384) [Added in 2022.10.20] 

Backbone |  Train DB  | Config | OneDrive | GDrive
:-|:-|:-|:-|:-
SwinB | DIS5K-TR | [InSPyReNet_SwinB_DIS5K.yaml](../configs/extra_dataset/InSPyReNet_SwinB_DIS5K.yaml) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERKrQ_YeoJRHl_3HcH8ZJLoBedsa6hZlmIIf66wobZRGuw?e=EywJmS) | [Link](https://drive.google.com/file/d/1Sj7GZoocGMHyKNhFnQQc1FTs76ysJIX3/view?usp=sharing)


## Pre-Computed Saliency Maps

Note: Due to the cloud memory shortage, we only provide results trained on DUTS-TR only. Please generate yourself for the models with extra training datasets if you need. 

Backbone | DUTS-TE | DUT-OMRON | ECSSD | HKU-IS | PASCAL-S | DAVIS-S | HRSOD-TE | UHRSD-TE
:-|:-|:-|:-|:-|:-|:-|:-|:-
Res2Net50 | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Eb0iKXGX1vxEjPhe9KGBKr0Bv7v2vv6Ua5NFybwc6aIi1w?e=oHnGyJ) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ef1HaYMvgh1EuuOL8bw3JGYB41-yo6KdTD8FGXcFZX3-Bg?e=TkW2m8) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdEQQ8o-yI9BtTpROcuB_iIBFSIk0uBJAkNyob0WI04-kw?e=cwEj2V) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ec6LyrumVZ9PoB2Af0OW4dcBrDht0OznnwOBYiu8pdyJ4A?e=Y04Fmn) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETPijMHlTRZIjqO5H4LBknUBmy8TGDwOyUQ1H4EnIpHVOw?e=k1afrh) | N/A | N/A | N/A |
SwinB | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETumLjuBantLim4kRqj4e_MBpK_X5XrTwjGQUToN8TKVjw?e=ZT8AWy) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZbwxhwT6dtHkBJrIMMjTnkBK_HaDTXgHcDSjxuswZKTZw?e=9XeE4b) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESfQK-557uZOmUwG5W49j0EBK42_7dMOaQcPsc_U1zsYlA?e=IvjkKX) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EURH96JUp55EgUHI0A8RzKoBBqvQc1nVb_a67RgwOY7f-w?e=IP9xKa) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EakMpwONph9EmnCM2rS3hn4B_TL42T6tuLjBEeEa5ndkIw?e=XksfA5) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETUCKFX0k8lAvpsDj5sT23QB2ohuE_ST7oQnWdaW7AoCIw?e=MbSmM2) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea6kf6Kk8fpIs15WWDfJMoYBeQUeo9WXvYx9oM5yWFE1Jg?e=RNN0Ns) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EVJLvAP3HwtHksZMUolIfCABHqP7GgAWcG_1V5T_Xrnr2g?e=ct3pzo) |

### * :new: DIS5K Results [Added in 2022.10.20]

Backbone | DIS-VD | DIS-TE1 | DIS-TE2 | DIS-TE3 | DIS-TE4
:-|:-|:-|:-|:-|:-|
SwinB | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUbzddb_QRRCtnXC8Xl6vZoBC6IqOfom52BWbzOYk-b2Ow?e=aqJYi1) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESeW_SOD26tHjBLymmgFaXwBIJlljzNycaGWXLpOp_d_kA?e=2EyMai) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYWT5fZDjI5Bn-iQM1TsB1num0-UqfJC1TIv-LuOXoA?e=jCcnty) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQXm1DEBfaNJmH0B-A3o23kBn4v5j53kP2nF9CpG9SQkyw?e=lEUiZh) | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZeH2ufGsFZIoUh6D8Rtv88BBF_ddQXav4xYXXRP_ayEAg?e=AMzIp8)
