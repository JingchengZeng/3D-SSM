import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

from dataset import create_cd_dataloader, create_cd_dataset
import models as Model
from models.loss import *
import core.logger as Logger
from misc.torchutils import get_scheduler, save_network
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
    opt = Logger.parse(args, "train")
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
    print("Creat [train] change-detection dataloader")
    train_set = create_cd_dataset(opt['datasets'], opt['phase'])
    train_loader = create_cd_dataloader(train_set, opt['datasets'], opt['phase'])
    opt['len_train_dataloader'] = len(train_loader)

    print("Creat [val] change-detection dataloader")
    val_set = create_cd_dataset(dataset_opt=opt['datasets'], phase="val")
    val_loader = create_cd_dataloader(val_set, opt['datasets'], "val")
    opt['len_val_dataloader'] = len(val_loader)

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

    #################
    # Training loop #
    #################
    metrics = (
        Precision(mode='accum'), 
        Recall(mode='accum'), 
        F1Score(mode='accum'), 
        Accuracy(mode='accum')
    )
    losses = Meter()
    train_loss = 0.0
    for current_epoch in range(epoch, opt['train']['n_epoch']):
        print("......Begin Training......\n")
        cd_model.train()

        #################
        #    Training   #
        #################
        message = 'lr: %0.7f' % optimer.param_groups[0]['lr']
        Logger.log_message(message)
        pbar = tqdm(enumerate(train_loader), 
                    desc=f"Train epoch: {current_epoch}/{opt['train']['n_epoch'] - 1}", 
                    unit="batch", total=len(train_loader))
        for current_step, train_data in pbar:
            train_im1 = train_data['A'].to(device)
            train_im2 = train_data['B'].to(device)
            pred_imgs, pred_img = cd_model(train_im1, train_im2)

            batch_size = train_im2.shape[0]
            gt = train_data['L'].to(device).long()
            for i in range(3):
                loss = loss_fun(pred_imgs[i], gt)
                train_loss += loss
            losses.update(train_loss.item(), n=batch_size)

            optimer.zero_grad()
            train_loss.backward()
            optimer.step()

            #pred score
            G_pred = pred_img.detach()
            G_pred = torch.argmax(G_pred, dim=1).cpu().numpy().astype('uint8')
            tar = train_data['L'].cpu().numpy().astype('uint8')
            for m in metrics:
                m.update(G_pred, tar, n=batch_size)

            pbar.set_postfix({
                'CD_loss': train_loss.item(),
                'losses.avg': losses.avg,
                'running_mf1': metrics[2].val
            })
            train_loss = 0.0

        ### log epoch status ###
        message = '[Training CD (epoch summary)]: epoch: [%d/%d]. epoch_F1=%.5f \n' % \
                    (current_epoch, opt['train']['n_epoch'] - 1, metrics[2].val)
        for m in metrics:
            message += " {} {:.4f}".format(m.__name__, m.val)
            m.reset()
        message += '\n'
        Logger.log_message(message)

        losses.reset()

        ##################
        ### validation ###
        ##################
        cd_model.eval()
        with torch.no_grad():
            pbar1 = tqdm(enumerate(val_loader), 
                    desc=f"Val epoch: {current_epoch}/{opt['train']['n_epoch'] - 1}", 
                    unit="batch", total=len(val_loader))
            for current_step, val_data in pbar1:
                val_img1 = val_data['A'].to(device)
                val_img2 = val_data['B'].to(device)
                _, pred_img = cd_model(val_img1, val_img2)

                batch_size = val_img2.shape[0]
                
                #pred score
                G_pred = pred_img.detach()
                G_pred = torch.argmax(G_pred, dim=1).cpu().numpy().astype('uint8')
                tar = val_data['L'].cpu().numpy().astype('uint8')
                for m in metrics:
                    m.update(G_pred, tar, n=batch_size)

                pbar1.set_postfix({
                    'Pre': metrics[0].val,
                    'Rec': metrics[1].val,
                    'F1 ': metrics[2].val,
                    'Acc': metrics[3].val
                })

            message = '[Test CD (epoch summary)]: epoch: [%d/%d]. epoch_F1=%.5f \n' % \
                    (current_epoch, opt['train']['n_epoch'] - 1, metrics[2].val)
            cF1 = metrics[2].val
            for m in metrics:
                message += " {} {:.4f}".format(m.__name__, m.val)
                m.reset()
            message += '\n'
            Logger.log_message(message)

            losses.reset()

            #best model
            if cF1 > best_F1:
                best_F1 = cF1
                is_best_model = True
                Logger.log_message('[Validation CD] Best model updated!!!!!')
                # save model
                save_network(opt, current_epoch, cd_model, optimer, best_F1, is_best_model)
            else:
                is_best_model = False
                Logger.log_message('[Validation CD] You need to keep training to get better results.')
                save_network(opt, current_epoch, cd_model, optimer, cF1, is_best_model)
            Logger.log_message('--- Proceed To The Next Epoch ----\n \n')

        get_scheduler(optimizer=optimer, args=opt['train']).step()
    Logger.log_message('End of training.')
