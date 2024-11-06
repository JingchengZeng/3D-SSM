import os
import json
import logging
from datetime import datetime
from collections import OrderedDict


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def parse(args, phase="train"):
    opt_path =args.config
    gpu_ids = args.gpu_ids
    save_path = args.save_path
    batch_size = args.batch_size
    pretrain = args.pretrain

    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
        #print(json_str)
    opt =json.loads(json_str, object_pairs_hook=OrderedDict)
    #print(opt)

    #create experiments folder
    experiments_root = os.path.join(
        save_path, '{}_{}_{}'.format(opt["name"], phase, get_timestamp()))
    for key, path in opt['path_cd'].items():
        opt['path_cd'][key] = os.path.join(experiments_root, path)
        mkdirs(opt['path_cd'][key])
    opt['path_cd']['experiments_root'] = experiments_root

    opt['phase'] = phase
    opt["resume"] = args.resume

    if pretrain == 'true':
        opt["model"]["pretrain"] = True
    elif pretrain == 'false':
        opt["model"]["pretrain"] = False
    elif pretrain is None:
        print(f'Pretrained Backbone load from "{opt["model"]["pretrain"]}"!!!!!')
    else:
        opt["model"]["pretrain"] = pretrain

    opt["datasets"]["batch_size"] = batch_size

    # export CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # print(f"gpu_id:{gpu_list}")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('expert CUDA_VISIBLE_DEVICES=' + gpu_list)
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    print(formatter)
    log_file = os.path.join(root, '{}.log'.format(phase))
    print(log_file)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def setup_logging(log_file_path):
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ])

def log_message(message):
    logging.info(message)
