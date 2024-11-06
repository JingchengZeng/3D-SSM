import core.logger as Logger
from models.Mamba3D import Mamba3D as mamba3d


def create_CD_model(opt):
    # Our CDMamba model

    if opt['model']['name'] == '3d-ssm':
        cd_model = mamba3d(pretrained=opt['model']['pretrain'], spatial_dims=opt['model']['spatial_dims'], 
                           init_filters=opt['model']['init_filters'], resdiual=opt['model']['resdiual'], 
                            in_channels=opt['model']['in_channels'], out_channels=opt['model']['n_classes'],
                            conv_mode=opt['model']['conv_mode'], up_mode=opt['model']['up_mode'], norm=opt['model']['norm'],
                            blocks_down=opt['model']['blocks_down'], blocks_up=opt['model']['blocks_up'],
                            diff_abs=opt['model']['diff_abs'], stage=opt['model']['stage'],
                            mamba_act=opt['model']['mamba_act'], local_query_model=opt['model']['local_query_model'])
    
    else:
        print("No model")
    Logger.log_message('CD Model [{:s}] is created.'.format(opt['model']['name']))
    return cd_model