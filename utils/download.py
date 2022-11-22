import os
import sys
import argparse

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

def download_and_unzip(filename, url, dest, unzip=True, **kwargs):
    if not os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
    
    os.system("wget -O {} {}".format(os.path.join(dest, filename), url))
    if unzip:
        os.system("unzip -o {} -d {}".format(os.path.join(dest, filename), dest))
        os.system("rm {}".format(os.path.join(dest, filename)))

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dest',  type=str, default=None)
    parser.add_argument('-e', '--extra', action='store_true', default=False)
    return parser.parse_args()

train_datasets = [
{'filename': "DUTS-TR.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQ7L2XS-5YFMuJGee7o7HQ8BdRSLO8utbC_zRrv-KtqQ3Q\?e\=bCSIeo\&download\=1", 'dest': "data/Train_Dataset", 'unzip': True, 'extra': False},
{'filename': "HRSOD-TR.zip" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUx92hUgZJNrWPj46PC0yEBXcorQskXOCSz8SnGH5AcLQ\?e\=WA5pc6\&download\=1", 'dest': "data/Train_Dataset", 'unzip': True, 'extra': True},
{'filename': "UHRSD-TR.zip" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea4_UCbsKmhKnMCccAJOTLgBmQFsQ4KhJSf2jx8WQqj3Wg\?e\=18kYZS\&download\=1", 'dest': "data/Train_Dataset", 'unzip': True, 'extra': True},
{'filename': "DIS-TR.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZtZJ493tVNJjBIpNLdus68B3u906PdWtHsf87pulj78jw\?e\=bUg2UQ\&download\=1", 'dest': "data/Train_Dataset", 'unzip': True, 'extra': True}
]

test_datasets = [
{'filename': "DUTS-TE.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfuCxjveXphPpIska9BxHDMBHpYroEKdVlq9HsonZ4wLDw\?e\=Mz5giA\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': False},
{'filename': "DUT-OMRON.zip" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERvApm9rHH5LiR4NJoWHqDoBCneUQNextk8EjQ_Hy0bUHg\?e\=wTRZQb\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': False},
{'filename': "ECSSD.zip"     , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ES_GCdS0yblBmnRaDZ8xmKQBPU_qeECTVB9vlPUups8bnA\?e\=POVAlG\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': False},
{'filename': "HKU-IS.zip"    , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYBRVvC1MJRAgSfzt0zaG94BU_UWaVrvpv4tjogu4vSV6w\?e\=TKN7hQ\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': False},
{'filename': "PASCAL-S.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUDGDckMnZHhEPy8YQGwBQB5MN3qInBkEygpIr7ccJdTQ\?e\=YarZaQ\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': False},
{'filename': "DAVIS-S.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ebam8I2o-tRJgADcq-r9YOkBCDyaAdWBVWyfN-xCYyAfDQ\?e\=Mqz8cK\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': False},
{'filename': "HRSOD-TE.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EbHOQZKC59xIpIdrM11ulWsBHRYY1wZY2njjWCDFXvT6IA\?e\=wls17m\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': False},
{'filename': "UHRSD-TE.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUpc8QJffNpNpESv-vpBi40BppucqOoXm_IaK7HYJkuOog\?e\=JTjGmS\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': False},
{'filename': "FSS-1000.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EaP6DogjMAVCtTKcC_Bx-YoBoBSWBo90lesVcMyuCN35NA\?e\=0DDohA\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': True},
{'filename': "MSRA-10K.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EauXsIkxBzhDjio6fW0TubUB4L7YJc0GMTaq7VfjI2nPsg\?e\=c5DIxg\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': True},
{'filename': "DIS-VD.zip"    , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYJm3BqheaxNhdVoMt6X41gBgVnE4dulBwkp6pbOQtcIrQ\?e\=T6dtXm\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': True},
{'filename': "DIS-TE1.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcGYE_Gc0cVHoHi_qUtmsawB_5v9RSpJS5PIAPlLu6xo9A\?e\=Nu5mkQ\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': True},
{'filename': "DIS-TE2.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdhgMdbZ049GvMv7tNrjbbQB1wL9Ok85YshiXIkgLyTfkQ\?e\=mPA6Po\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': True},
{'filename': "DIS-TE3.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcxXYC_3rXxKsQBrp6BdNb4BOKxBK3_vsR9RL76n7YVG-g\?e\=2M0cse\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': True},
{'filename': "DIS-TE4.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdkG2SUi8flJvoYbHHOmvMABsGhkCJCsLLZlaV2E_SZimA\?e\=zlM2kC\&download\=1", 'dest': "data/Test_Dataset", 'unzip': True, 'extra': True}
]

backbone_ckpts = [
{'filename': "res2net50_v1b_26w_4s-3cf99910.pth"         , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUO7GDBwoC9CulTPdnq_yhQBlc0SIyyELMy3OmrNhOjcGg\?e\=T3PVyG\&download\=1", 'dest': "data/backbone_ckpt", 'unzip': False, 'extra':False},
{'filename': "swin_base_patch4_window12_384_22kto1k.pth" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESlYCLy0endMhcZm9eC2A4ABatxupp4UPh03EcqFjbtSRw\?e\=7y6lLt\&download\=1", 'dest': "data/backbone_ckpt", 'unzip': False, 'extra':False}
]

pretrained_ckpts = [
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERqm7RPeNBFPvVxkA5P5G2AB-mtFsiYkCNHnBf0DcwpFzw\?e\=nayVno\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50"      , 'unzip': False, 'extra':False},   
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EV0ow4E8LddCgu5tAuAkMbcBpBYoEDmJgQg5wkiuvLoQUA\?e\=cOZspv\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB"          , 'unzip': False, 'extra':False}, 
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EWxPZoIKALlGsfrNgUFNvxwBC8IE8jzzhPNtzcbHmTNFcg\?e\=e22wmy\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR"    , 'unzip': False, 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQe-iy0AZctIkgl3o-BmVYUBn795wvii3tsnBq1fNUbc9g\?e\=gMZ4PV\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR"    , 'unzip': False, 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfsCbnfAU1RAqCJIkj1ewRgBhFetStsGB6SMSq_UJZimjA\?e\=Ghuacy\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR"   , 'unzip': False, 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EW2Qg-tMBBxNkygMj-8QgMUBiqHox5ExTOJl0LGLsn6AtA\?e\=Mam8Ur\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH"       , 'unzip': False, 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EeE8nnCt_AdFvxxu0JsxwDgBCtGchuUka6DW9za_epX-Qw\?e\=U7wZu9\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU"       , 'unzip': False, 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESKuh1zhToVFsIxhUUsgkbgBnu2kFXCFLRuSz1xxsKzjhA\?e\=02HDrm\&download\=1", 'dest': "snapshots/Plus_Ultra_LR"             , 'unzip': False, 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ET0R-yM8MfVHqI4g94AlL6AB-D6LxNajaWeDV4xbVQyh7w\?e\=l4JkZn\&download\=1", 'dest': "snapshots/Plus_Ultra"                , 'unzip': False, 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERKrQ_YeoJRHl_3HcH8ZJLoBedsa6hZlmIIf66wobZRGuw\?e\=EywJmS\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K_LR" , 'unzip': False, 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EShRbD-jZuRJiWv6DS2Us34BwgazGZvK1t4uTKvgE5379Q\?e\=8oVpS8\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K"    , 'unzip': False, 'extra':True}
]

precomputed_maps = [
{'filename': "DUTS-TE.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Eb0iKXGX1vxEjPhe9KGBKr0Bv7v2vv6Ua5NFybwc6aIi1w\?e\=oHnGyJ\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",       'unzip': True, 'extra': False},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ef1HaYMvgh1EuuOL8bw3JGYB41-yo6KdTD8FGXcFZX3-Bg\?e\=TkW2m8\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",       'unzip': True, 'extra': False},
{'filename': "ECSSD.zip"    , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdEQQ8o-yI9BtTpROcuB_iIBFSIk0uBJAkNyob0WI04-kw\?e\=cwEj2V\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",       'unzip': True, 'extra': False},
{'filename': "HKU-IS.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ec6LyrumVZ9PoB2Af0OW4dcBrDht0OznnwOBYiu8pdyJ4A\?e\=Y04Fmn\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",       'unzip': True, 'extra': False},
{'filename': "PASCAL-S.zip" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETPijMHlTRZIjqO5H4LBknUBmy8TGDwOyUQ1H4EnIpHVOw\?e\=k1afrh\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",       'unzip': True, 'extra': False},
{'filename': "DUTS-TE.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETumLjuBantLim4kRqj4e_MBpK_X5XrTwjGQUToN8TKVjw\?e\=ZT8AWy\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",           'unzip': True, 'extra': False},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZbwxhwT6dtHkBJrIMMjTnkBK_HaDTXgHcDSjxuswZKTZw\?e\=9XeE4b\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",           'unzip': True, 'extra': False},
{'filename': "ECSSD.zip",     'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESfQK-557uZOmUwG5W49j0EBK42_7dMOaQcPsc_U1zsYlA\?e\=IvjkKX\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",           'unzip': True, 'extra': False},
{'filename': "HKU-IS.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EURH96JUp55EgUHI0A8RzKoBBqvQc1nVb_a67RgwOY7f-w\?e\=IP9xKa\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",           'unzip': True, 'extra': False},
{'filename': "PASCAL-S.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EakMpwONph9EmnCM2rS3hn4B_TL42T6tuLjBEeEa5ndkIw\?e\=XksfA5\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",           'unzip': True, 'extra': False},
{'filename': "DAVIS-S.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETUCKFX0k8lAvpsDj5sT23QB2ohuE_ST7oQnWdaW7AoCIw\?e\=MbSmM2\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",           'unzip': True, 'extra': False},
{'filename': "HRSOD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea6kf6Kk8fpIs15WWDfJMoYBeQUeo9WXvYx9oM5yWFE1Jg\?e\=RNN0Ns\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",           'unzip': True, 'extra': False},
{'filename': "UHRSD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EVJLvAP3HwtHksZMUolIfCABHqP7GgAWcG_1V5T_Xrnr2g\?e\=ct3pzo\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",           'unzip': True, 'extra': False},
{'filename': "DUTS-TE.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EbaXjDWFb6lGp9x5ae9mJPgBt9dkmgclq9XrXDjj4B5qSw\?e\=57wnhE\&download&=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR",     'unzip': True, 'extra': True},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EU7uNIpKPXZHrDNt3gapGfsBaUrkCj67-Paj4w_7E8xs1g\?e\=n0QBR1\&download&=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR",     'unzip': True, 'extra': True},
{'filename': "ECSSD.zip",     'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EeBq5uU02FRLiW-0g-mQ_CsB3iHlMgAyOSY2Deu5suo9pQ\?e\=zjhB33\&download&=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR",     'unzip': True, 'extra': True},
{'filename': "HKU-IS.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EY5AppwMB4hFqbISFL_u5QMBeux7dQWtDeXaMMcAyLqLqQ\?e\=N71XVt\&download&=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR",     'unzip': True, 'extra': True},
{'filename': "PASCAL-S.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfhejUtLallLroU6jjgDl-oBSHITWnkdiU6NVV95DO5YqQ\?e\=T6rrRW\&download&=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR",     'unzip': True, 'extra': True},
{'filename': "DAVIS-S.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EesN4fRAE4JJk1aZ_tH3EBQBgirysALcgfw1Ipsa9dLe9Q\?e\=b5oWsg\&download&=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR",     'unzip': True, 'extra': True},
{'filename': "HRSOD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETpVYDijnZhInN9zBHRuwhUBSPslqe9FP0m3Eo3TWS0d5A\?e\=QTzfx6\&download&=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR",     'unzip': True, 'extra': True},
{'filename': "UHRSD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUkKVH3LSyFLq6UXuvWEVuYBEH8W8uAKgyolLRVuIUILag\?e\=y1SceD\&download&=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR",     'unzip': True, 'extra': True},
{'filename': "DUTS-TE.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUSblgWAwg9Plc4LCGj4TLwB-7HLEdZGJqF1jHOU55g3OA\?e\=2YT3zM\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR",     'unzip': True, 'extra': True},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESNaVX4Fh5JHn5jOfgnSWi4Bx1bc9t6pg79IoG3mrpZpAw\?e\=M8D0CM\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR",     'unzip': True, 'extra': True},
{'filename': "ECSSD.zip",     'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETKjTH1vcVZDu5ahRaw4cb8B7JKaPMR-0Uae1DbwarobIA\?e\=Qw67IZ\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR",     'unzip': True, 'extra': True},
{'filename': "HKU-IS.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcC9-lPgAXZHs7th9DiVjygB-zPIq_1Ii6i1GpbLGc1iPQ\?e\=EVXKp9\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR",     'unzip': True, 'extra': True},
{'filename': "PASCAL-S.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EXjyAS4XyzlPqrpaizgElioBdmgd4E81qQzj11Qm4xo5sA\?e\=hoOzc2\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR",     'unzip': True, 'extra': True},
{'filename': "DAVIS-S.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcSCMR033GVNotOilIzYhIsBikzb8ZzGlkuW6aSNMlUpqQ\?e\=TFcgvE\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR",     'unzip': True, 'extra': True},
{'filename': "HRSOD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUHRSEANcmVFjcS_K-PeYr0B6VPXPgb2AHFUnlJYrf3dOQ\?e\=unwcqV\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR",     'unzip': True, 'extra': True},
{'filename': "UHRSD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EU2gb0hS5kZBgKNXWqQAomsBXU-zjGCXKAzYNNk4d6EAiQ\?e\=pjhiN2\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR",     'unzip': True, 'extra': True},
{'filename': "DUTS-TE.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ee7Y647ERzxMgBoFceEEO6kBIUkIlmYHoizMj71gT37sxw\?e\=xDt83C\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR",    'unzip': True, 'extra': True},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESFGN3FCdzRAvlsW6bEaGdoBYNoJgK4DAjaS6WkVVyI_QQ\?e\=nYHklV\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR",    'unzip': True, 'extra': True},
{'filename': "ECSSD.zip",     'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EeONq5kOirRCkErrb6fFqd8B4w4SMZXBY1Q2mJvZcRsGdQ\?e\=K7fwQt\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR",    'unzip': True, 'extra': True},
{'filename': "HKU-IS.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea5P4CBBQatPiUzsH53lckoB0k23haePzuERBfyJXaCbBg\?e\=AZ96mc\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR",    'unzip': True, 'extra': True},
{'filename': "PASCAL-S.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea2pezB7eo1BraeBpIA8YZoBkVY38rRa3KrwSIzY1cn2dQ\?e\=o121S6\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR",    'unzip': True, 'extra': True},
{'filename': "DAVIS-S.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfNDzH8O54pGtnAxit_hUjUBK9poVq4sxxnJjSG7PUQCkw\?e\=OWt7k8\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR",    'unzip': True, 'extra': True},
{'filename': "HRSOD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESfZaII0pO9IqqL1FkpjIuAB8SGxLcslJeWTuKQxPNFIVA\?e\=Ce1CWg\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR",    'unzip': True, 'extra': True},
{'filename': "UHRSD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYsQqf1GKShJhBnkZ6gD5PABPOcjRcUGSfvTbe-wYh2O2Q\?e\=t4Xlxv\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR",    'unzip': True, 'extra': True},
{'filename': "DUTS-TE.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EWNaRqtzhtNFhMVfLcoyfqQBw35M8q8bxME3yZyhkTtc7Q\?e\=jrJe3v\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH",        'unzip': True, 'extra': True},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZF5_s8JfR9HqGBUZHSM_j4BVVMONp38_gJ1ekEdvlM-qQ\?e\=0chMdl\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH",        'unzip': True, 'extra': True},
{'filename': "ECSSD.zip",     'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZuaFObyNOtKg0W5cM7bqPYBZYGg7Z3V3i4sClI6bU_ntA\?e\=BxxQI7\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH",        'unzip': True, 'extra': True},
{'filename': "HKU-IS.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EY-yEnNNNT5KpiURhiDAkDEBMkiA1QwQ_T0wB1UC75GXVg\?e\=Lle02B\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH",        'unzip': True, 'extra': True},
{'filename': "PASCAL-S.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYfbsgWhm7lAlX_wj_WZZowBV_-l-UvvThC4LJEKpV0BQQ\?e\=zTiKpI\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH",        'unzip': True, 'extra': True},
{'filename': "DAVIS-S.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ef0GUP7c0bBBonHlqgB988YB0rgxCFq3oo0u8xCN8wfyyQ\?e\=LCb8UV\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH",        'unzip': True, 'extra': True},
{'filename': "HRSOD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EVUZBnRpa35AmrvdUybsQDMBzMZvuJWe5tT7635lh9MHDQ\?e\=FlpQW1\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH",        'unzip': True, 'extra': True},
{'filename': "UHRSD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETfZ_zdrDvhOh21u2mqVhigBSxn3vlfKVIwXhRfzzSSFzA\?e\=kXBBi9\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH",        'unzip': True, 'extra': True},
{'filename': "DUTS-TE.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZq4JUACKCBMk2bn4yoWz6sBOKrSFTPfL7d5xopc1uDw_A\?e\=RtVHSl\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU",        'unzip': True, 'extra': True},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETJaqoSPaYtNkc8eSGDeKzMBbjbuOAWgJwG4q52bW87aew\?e\=Pguh4b\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU",        'unzip': True, 'extra': True},
{'filename': "ECSSD.zip",     'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZAeCI6BPMdNsicnQ-m1pVEBwAhOiIcbelhOMoRGXGEvVA\?e\=BQKd7Q\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU",        'unzip': True, 'extra': True},
{'filename': "HKU-IS.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EVmGvZGz54JOvrIymLsSwq4Bpos3vWSXZm3oV7-qmGZgHA\?e\=4UhDgv\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU",        'unzip': True, 'extra': True},
{'filename': "PASCAL-S.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERHDUybOh4ZKkqWZpcu7MiMBFuTK6wACkKUZaNeEQGbCNQ\?e\=GCQnoe\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU",        'unzip': True, 'extra': True},
{'filename': "DAVIS-S.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESPmZXTnfO5CrCoo_0OADxgBt_3FoU5mSFoSE4QWbWxumQ\?e\=HAsAYz\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU",        'unzip': True, 'extra': True},
{'filename': "HRSOD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdTnUwEeMZNBrPSBBbGZKQcBmVshSTfca9qz_BqNpAUpOg\?e\=HsJ4Gx\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU",        'unzip': True, 'extra': True},
{'filename': "UHRSD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ET48owfVQEdImrh0V4gx8_ABsYXgbIJqtpq77aK_U28VwQ\?e\=h8er3H\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU",        'unzip': True, 'extra': True},
{'filename': "DIS-VD.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUbzddb_QRRCtnXC8Xl6vZoBC6IqOfom52BWbzOYk-b2Ow\?e\=aqJYi1\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K_LR",  'unzip': True, 'extra': True},
{'filename': "DIS-TE1.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESeW_SOD26tHjBLymmgFaXwBIJlljzNycaGWXLpOp_d_kA\?e\=2EyMai\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K_LR",  'unzip': True, 'extra': True},
{'filename': "DIS-TE2.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYWT5fZDjI5Bn-lr-iQM1TsB1num0-UqfJC1TIv-LuOXoA\?e\=jCcnty\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K_LR",  'unzip': True, 'extra': True},
{'filename': "DIS-TE3.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQXm1DEBfaNJmH0B-A3o23kBn4v5j53kP2nF9CpG9SQkyw\?e\=lEUiZh\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K_LR",  'unzip': True, 'extra': True},
{'filename': "DIS-TE4.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZeH2ufGsFZIoUh6D8Rtv88BBF_ddQXav4xYXXRP_ayEAg\?e\=AMzIp8\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K_LR",  'unzip': True, 'extra': True},
{'filename': "DIS-VD.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUisiphB_W5BjgfpIYu9oNgB_fY4XxL-MhR2gR-ZZUt49Q\?e\=gqorYs\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K",     'unzip': True, 'extra': True},
{'filename': "DIS-TE1.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EW2y54ROYIZFlEq5ilRFwOsBSrIm2-HGsUHPHykaJvUBfA\?e\=797fxr\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K",     'unzip': True, 'extra': True},
{'filename': "DIS-TE2.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ER6yEGZkgWVOsL-mauYgPyoBDIoU0Mck-twEBgQi5g3Mxw\?e\=0yJZTT\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K",     'unzip': True, 'extra': True},
{'filename': "DIS-TE3.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Edvzj0iZ8hdDthm4Q-p2YHgBhP1X5z4AAccAoUasr2nihA\?e\=1dognG\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K",     'unzip': True, 'extra': True},
{'filename': "DIS-TE4.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EWwMORYg8DlCgGGqOQFThZ8BgIU9wV9-0DwLrKldQl7N8w\?e\=nCwuqy\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K",     'unzip': True, 'extra': True},
]

if __name__ == '__main__':
    args = _args()
    
    if args.dest is not None:
        os.system('ln -s ' + os.path.join(args.dest, 'data', 'Train_Dataset')  + ' ' + os.path.join(repopath, 'data'))
        os.system('ln -s ' + os.path.join(args.dest, 'data', 'Test_Dataset')   + ' ' + os.path.join(repopath, 'data'))
        os.system('ln -s ' + os.path.join(args.dest, 'data', 'backbone_ckpt')  + ' ' + os.path.join(repopath, 'data'))
        os.system('ln -s ' + os.path.join(args.dest, 'snapshots') + ' ' + repopath)
    else:
        args.dest = repopath
        os.makedirs(os.path.join(args.dest, 'data', 'Train_Dataset'), exist_ok=True)
        os.makedirs(os.path.join(args.dest, 'data', 'Test_Dataset'),  exist_ok=True)
        os.makedirs(os.path.join(args.dest, 'data', 'backbone_ckpt'), exist_ok=True)
        os.makedirs(os.path.join(args.dest, 'snapshots'), exist_ok=True)
    
    download_list = train_datasets + test_datasets + backbone_ckpts + pretrained_ckpts + precomputed_maps
    
    for dinfo in download_list:
        dinfo['dest'] = os.path.join(args.dest, dinfo['dest'])
        if not dinfo['extra'] or args.extra:
            download_and_unzip(**dinfo)