import os
import time
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.optim as optim

from dataset import create_cd_dataloader, create_cd_dataset
import models as Model
from models.loss import *
import core.logger as Logger
from misc.metric_tools import (Meter, Precision, Recall, Accuracy, F1Score)


if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        default='configs/3dssm+whu+train.json',
                        help='JSON file for configuration')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--pretrain', type=str, help='Pretrianed model for Backbone')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_path', type=str, default='experiments')

    #paser config
    args =parser.parse_args()
    opt = Logger.parse(args, "test")
    #Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # Set random seed
    RNG_SEED = 952469
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    torch.cuda.manual_seed(RNG_SEED)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    #logging
    Logger.setup_logging(os.path.join(opt['path_cd']['log'], opt['phase'] + '.txt'))
    Logger.log_message(Logger.dict2str(opt))

    #dataset
    print("Creat [test] change-detection dataloader")
    test_set = create_cd_dataset(dataset_opt=opt['datasets'], phase="test")
    test_loader = create_cd_dataloader(test_set, opt['datasets'], "test")
    opt['len_test_dataloader'] = len(test_loader)

    Logger.log_message('Initial Dataset Finished')

    #Create cd model
    cd_model = Model.create_CD_model(opt)
        
    #Create criterion
    if opt['train']['loss'] == 'ce_dice':
        loss_fun = ce_dice
    elif opt['train']['loss'] == 'ce':
        loss_fun = cross_entropy
    elif opt['train']['loss'] == 'dice':
        loss_fun = dice
    elif opt['train']['loss'] == 'ce2_dice1':
        loss_fun = ce2_dice1
    elif opt['train']['loss'] == 'ce1_dice2':
        loss_fun = ce1_dice2

    #Create optimer
    if opt['train']["optimizer"]["type"] == 'adam':
        optimer = optim.Adam(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimer = optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimer = optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                            momentum=0.9, weight_decay=5e-4)
        
    # resume
    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
    load_path = opt["resume"]
    epoch = 0
    best_F1 = 0.0
    if load_path is not None:
        checkpoint = torch.load(load_path)
        epoch = checkpoint['epoch'] + 1
        best_F1 = checkpoint['F1']
        cd_model.load_state_dict(checkpoint['model'], strict=False)

        optimer.load_state_dict(checkpoint['optimizer'])
        for state in optimer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    cd_model.to(device)

    Logger.log_message('Begin Model Evaluation (testing).')
    test_result_path = '{}/test/'.format(opt['path_cd']['result'])
    os.makedirs(test_result_path, exist_ok=True)

    metrics = (
        Precision(mode='accum'), 
        Recall(mode='accum'), 
        F1Score(mode='accum'), 
        Accuracy(mode='accum')
    )
    
    cd_model.eval()
    total_time = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), 
                    desc="Testing", 
                    unit="batch", total=len(test_loader))
        for current_step, test_data in pbar:
            test_img1 = test_data['A'].to(device)
            test_img2 = test_data['B'].to(device)
            
            start_time = time.time()
            _, pred_img = cd_model(test_img1, test_img2)
            inference_time = time.time() - start_time
            total_time += inference_time

            batch_size = test_img2.shape[0]
            gt = test_data['L'].cpu().numpy().astype('uint8')
            G_pred = pred_img.detach()
            G_pred = torch.argmax(G_pred, dim=1).cpu().numpy().astype('uint8')
            for m in metrics:
                m.update(G_pred, gt, n = batch_size)
            
            pbar.set_postfix({
                'Pre': metrics[0].val,
                'Rec': metrics[1].val,
                'F1 ': metrics[2].val,
                'Acc': metrics[3].val
            })
            if args.save:
                for i in range(G_pred.shape[0]):
                    img = Image.fromarray(G_pred[i] * 255)  # 将二值图转换为0和255的灰度图
                    img.save(os.path.join(test_result_path, test_data['Name'][i]))

        average_time = total_time / (len(test_loader) * args.batch_size)
        print("num:{}".format(len(test_loader) * args.batch_size))
        print('Average inference time per image: {:.5f} seconds\n'.format(average_time))
        message = '[Test CD summary]: Test mF1=%.5f \n' % \
                    (metrics[2].val)
        for m in metrics:
            message += "{} {:.4f} ".format(m.__name__, m.val)
            m.reset()
        message += '\n'
        Logger.log_message(message)
        Logger.log_message('End of testing...')
