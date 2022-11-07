import os
import sys
import argparse

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

def download_and_unzip(filename, url, dest, **kwargs):
    if not os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
    
    os.system("wget -O {} {}".format(os.path.join(dest, filename), url))
    os.system("unzip {} -d {}".format(os.path.join(dest, filename), dest))
    os.system("rm {}".format(os.path.join(dest, filename)))

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dest',  type=str, default=None)
    parser.add_argument('-e', '--extra', action='store_true', default=False)
    return parser.parse_args()

train_datasets = [
{'filename': "DUTS-TR.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQ7L2XS-5YFMuJGee7o7HQ8BdRSLO8utbC_zRrv-KtqQ3Q\?e\=bCSIeo\&download\=1", 'dest': "data/Train_Dataset", 'extra': False},
{'filename': "HRSOD-TR.zip" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUx92hUgZJNrWPj46PC0yEBXcorQskXOCSz8SnGH5AcLQ\?e\=WA5pc6\&download\=1", 'dest': "data/Train_Dataset", 'extra': True},
{'filename': "UHRSD-TR.zip" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea4_UCbsKmhKnMCccAJOTLgBmQFsQ4KhJSf2jx8WQqj3Wg\?e\=18kYZS\&download\=1", 'dest': "data/Train_Dataset", 'extra': True},
{'filename': "DIS-TR.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZtZJ493tVNJjBIpNLdus68B3u906PdWtHsf87pulj78jw\?e\=bUg2UQ\&download\=1", 'dest': "data/Train_Dataset", 'extra': True}
]

test_datasets = [
{'filename': "DUTS-TE.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfuCxjveXphPpIska9BxHDMBHpYroEKdVlq9HsonZ4wLDw\?e\=Mz5giA\&download\=1", 'dest': "data/Test_Dataset", 'extra': False},
{'filename': "DUT-OMRON.zip" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERvApm9rHH5LiR4NJoWHqDoBCneUQNextk8EjQ_Hy0bUHg\?e\=wTRZQb\&download\=1", 'dest': "data/Test_Dataset", 'extra': False},
{'filename': "ECSSD.zip"     , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ES_GCdS0yblBmnRaDZ8xmKQBPU_qeECTVB9vlPUups8bnA\?e\=POVAlG\&download\=1", 'dest': "data/Test_Dataset", 'extra': False},
{'filename': "HKU-IS.zip"    , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYBRVvC1MJRAgSfzt0zaG94BU_UWaVrvpv4tjogu4vSV6w\?e\=TKN7hQ\&download\=1", 'dest': "data/Test_Dataset", 'extra': False},
{'filename': "PASCAL-S.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUDGDckMnZHhEPy8YQGwBQB5MN3qInBkEygpIr7ccJdTQ\?e\=YarZaQ\&download\=1", 'dest': "data/Test_Dataset", 'extra': False},
{'filename': "DAVIS-S.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ebam8I2o-tRJgADcq-r9YOkBCDyaAdWBVWyfN-xCYyAfDQ\?e\=Mqz8cK\&download\=1", 'dest': "data/Test_Dataset", 'extra': False},
{'filename': "HRSOD-TE.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EbHOQZKC59xIpIdrM11ulWsBHRYY1wZY2njjWCDFXvT6IA\?e\=wls17m\&download\=1", 'dest': "data/Test_Dataset", 'extra': False},
{'filename': "UHRSD-TE.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUpc8QJffNpNpESv-vpBi40BppucqOoXm_IaK7HYJkuOog\?e\=JTjGmS\&download\=1", 'dest': "data/Test_Dataset", 'extra': False},
{'filename': "DIS-VD.zip"    , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYJm3BqheaxNhdVoMt6X41gBgVnE4dulBwkp6pbOQtcIrQ\?e\=T6dtXm\&download\=1", 'dest': "data/Test_Dataset", 'extra': True},
{'filename': "DIS-TE1.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcGYE_Gc0cVHoHi_qUtmsawB_5v9RSpJS5PIAPlLu6xo9A\?e\=Nu5mkQ\&download\=1", 'dest': "data/Test_Dataset", 'extra': True},
{'filename': "DIS-TE2.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdhgMdbZ049GvMv7tNrjbbQB1wL9Ok85YshiXIkgLyTfkQ\?e\=mPA6Po\&download\=1", 'dest': "data/Test_Dataset", 'extra': True},
{'filename': "DIS-TE3.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcxXYC_3rXxKsQBrp6BdNb4BOKxBK3_vsR9RL76n7YVG-g\?e\=2M0cse\&download\=1", 'dest': "data/Test_Dataset", 'extra': True},
{'filename': "DIS-TE4.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdkG2SUi8flJvoYbHHOmvMABsGhkCJCsLLZlaV2E_SZimA\?e\=zlM2kC\&download\=1", 'dest': "data/Test_Dataset", 'extra': True}
]

backbone_ckpts = [
{'filename': "res2net50_v1b_26w_4s-3cf99910.pth"         , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUO7GDBwoC9CulTPdnq_yhQBlc0SIyyELMy3OmrNhOjcGg\?e\=T3PVyG\&download\=1", 'dest': "data/backbone_ckpt", 'extra':False},
{'filename': "swin_base_patch4_window12_384_22kto1k.pth" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESlYCLy0endMhcZm9eC2A4ABatxupp4UPh03EcqFjbtSRw\?e\=7y6lLt\&download\=1", 'dest': "data/backbone_ckpt", 'extra':False}
]

pretrained_ckpts = [
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERqm7RPeNBFPvVxkA5P5G2AB-mtFsiYkCNHnBf0DcwpFzw\?e\=nayVno\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50"   , 'extra':True},   
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EV0ow4E8LddCgu5tAuAkMbcBpBYoEDmJgQg5wkiuvLoQUA\?e\=cOZspv\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB"       , 'extra':True}, 
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EWxPZoIKALlGsfrNgUFNvxwBC8IE8jzzhPNtzcbHmTNFcg\?e\=e22wmy\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH_LR" , 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQe-iy0AZctIkgl3o-BmVYUBn795wvii3tsnBq1fNUbc9g\?e\=gMZ4PV\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU_LR" , 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfsCbnfAU1RAqCJIkj1ewRgBhFetStsGB6SMSq_UJZimjA\?e\=Ghuacy\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DHU_LR", 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EW2Qg-tMBBxNkygMj-8QgMUBiqHox5ExTOJl0LGLsn6AtA\?e\=Mam8Ur\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DH"    , 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EeE8nnCt_AdFvxxu0JsxwDgBCtGchuUka6DW9za_epX-Qw\?e\=U7wZu9\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_HU"    , 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESKuh1zhToVFsIxhUUsgkbgBnu2kFXCFLRuSz1xxsKzjhA\?e\=02HDrm\&download\=1", 'dest': "snapshots/Plus_Ultra"             , 'extra':True},
{'filename': "latest.pth", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERKrQ_YeoJRHl_3HcH8ZJLoBedsa6hZlmIIf66wobZRGuw\?e\=EywJmS\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K" , 'extra':True}
]

precomputed_maps = [
{'filename': "DUTS-TE.zip"  , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Eb0iKXGX1vxEjPhe9KGBKr0Bv7v2vv6Ua5NFybwc6aIi1w\?e\=oHnGyJ\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",   'extra': False},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ef1HaYMvgh1EuuOL8bw3JGYB41-yo6KdTD8FGXcFZX3-Bg\?e\=TkW2m8\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",   'extra': False},
{'filename': "ECSSD.zip"    , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdEQQ8o-yI9BtTpROcuB_iIBFSIk0uBJAkNyob0WI04-kw\?e\=cwEj2V\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",   'extra': False},
{'filename': "HKU-IS.zip"   , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ec6LyrumVZ9PoB2Af0OW4dcBrDht0OznnwOBYiu8pdyJ4A\?e\=Y04Fmn\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",   'extra': False},
{'filename': "PASCAL-S.zip" , 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETPijMHlTRZIjqO5H4LBknUBmy8TGDwOyUQ1H4EnIpHVOw\?e\=k1afrh\&download\=1", 'dest': "snapshots/InSPyReNet_Res2Net50",   'extra': False},
{'filename': "DUTS-TE.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETumLjuBantLim4kRqj4e_MBpK_X5XrTwjGQUToN8TKVjw\?e\=ZT8AWy\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",       'extra': False},
{'filename': "DUT-OMRON.zip", 'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZbwxhwT6dtHkBJrIMMjTnkBK_HaDTXgHcDSjxuswZKTZw\?e\=9XeE4b\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",       'extra': False},
{'filename': "ECSSD.zip",     'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESfQK-557uZOmUwG5W49j0EBK42_7dMOaQcPsc_U1zsYlA\?e\=IvjkKX\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",       'extra': False},
{'filename': "HKU-IS.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EURH96JUp55EgUHI0A8RzKoBBqvQc1nVb_a67RgwOY7f-w\?e\=IP9xKa\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",       'extra': False},
{'filename': "PASCAL-S.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EakMpwONph9EmnCM2rS3hn4B_TL42T6tuLjBEeEa5ndkIw\?e\=XksfA5\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",       'extra': False},
{'filename': "DAVIS-S.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ETUCKFX0k8lAvpsDj5sT23QB2ohuE_ST7oQnWdaW7AoCIw\?e\=MbSmM2\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",       'extra': False},
{'filename': "HRSOD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea6kf6Kk8fpIs15WWDfJMoYBeQUeo9WXvYx9oM5yWFE1Jg\?e\=RNN0Ns\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",       'extra': False},
{'filename': "UHRSD-TE.zip",  'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EVJLvAP3HwtHksZMUolIfCABHqP7GgAWcG_1V5T_Xrnr2g\?e\=ct3pzo\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB",       'extra': False},
{'filename': "DIS-VD.zip",    'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUbzddb_QRRCtnXC8Xl6vZoBC6IqOfom52BWbzOYk-b2Ow\?e\=aqJYi1\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K", 'extra': True},
{'filename': "DIS-TE1.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESeW_SOD26tHjBLymmgFaXwBIJlljzNycaGWXLpOp_d_kA\?e\=2EyMai\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K", 'extra': True},
{'filename': "DIS-TE2.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYWT5fZDjI5Bn-lr-iQM1TsB1num0-UqfJC1TIv-LuOXoA\?e\=jCcnty\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K", 'extra': True},
{'filename': "DIS-TE3.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQXm1DEBfaNJmH0B-A3o23kBn4v5j53kP2nF9CpG9SQkyw\?e\=lEUiZh\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K", 'extra': True},
{'filename': "DIS-TE4.zip",   'url': "https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZeH2ufGsFZIoUh6D8Rtv88BBF_ddQXav4xYXXRP_ayEAg\?e\=AMzIp8\&download\=1", 'dest': "snapshots/InSPyReNet_SwinB_DIS5K", 'extra': True},
]

if __name__ == '__main__':
    args = _args()
    os.makedirs(os.path.join(repopath, 'data', 'Train_Dataset'), exist_ok=True)
    os.makedirs(os.path.join(repopath, 'data', 'Test_Dataset'),  exist_ok=True)
    os.makedirs(os.path.join(repopath, 'data', 'backbone_ckpt'), exist_ok=True)
    os.makedirs(os.path.join(repopath, 'snapshots'), exist_ok=True)
    
    if args.dest is not None:
        os.system('ln -s ' + os.path.join(args.dest, 'data', 'Train_Dataset')  + ' ' + os.path.join(repopath, 'data', 'Train_Dataset'))
        os.system('ln -s ' + os.path.join(args.dest, 'data', 'Test_Dataset')   + ' ' + os.path.join(repopath, 'data', 'Test_Dataset'))
        os.system('ln -s ' + os.path.join(args.dest, 'data', 'backbone_ckpt')  + ' ' + os.path.join(repopath, 'data', 'backbone_ckpt'))
        os.system('ln -s ' + os.path.join(args.dest, 'snapshots') + ' ' + os.path.join(repopath, 'snapshots'))
    
    download_list = train_datasets + test_datasets + backbone_ckpts + pretrained_ckpts + precomputed_maps
    
    for dinfo in download_list:
        if args.dest is None:
            dinfo['dest'] = os.path.join(repopath,  dinfo['dest'])
        else:
            dinfo['dest'] = os.path.join(args.dest, dinfo['dest'])
        
        if not dinfo['extra'] or args.extra:
            download_and_unzip(**dinfo)
            break