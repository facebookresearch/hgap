import time
import sys
import os
import glob
import pickle
import json
import torch
import pdb

def mkdir(savepath, prune_fname=False):
    """
        returns `True` iff `savepath` is created
    """
    if prune_fname:
        savepath = os.path.dirname(savepath)
    if not os.path.exists(savepath):
        try:
            os.makedirs(savepath)
        except:
            print(f'[ utils/serialization ] Warning: did not make directory: {savepath}')
            return False
        return True
    else:
        return False

def get_latest_epoch(loadpath, prior='', debug=False):
    states = glob.glob1(loadpath, prior+'state_*')
    states = [s for s in states if 'running' not in s]
    if debug:
        states = [s for s in states if 'debug' in s]
        debug_suffx = '_debug'
    else:
         states = [s for s in states if 'debug' not in s]
         debug_suffx = ''
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace(debug_suffx, '').replace(prior+'state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_transformer_model(logger, *loadpath, epoch=None, device='cuda:0', debug=False, type='prior'):
    loadpath = os.path.join(*loadpath)
    config_path = os.path.join(loadpath, f'{type}_model_config.pkl')
    debug_suffix = '' if not debug else '_debug'

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath, f"{type}_", debug)

    logger.debug(f'[ utils/serialization ] Loading model epoch: {epoch}', main_process_only=True)
    state_path = os.path.join(loadpath, f'{type}_state_{epoch}{debug_suffix}.pt')

    config = pickle.load(open(config_path, 'rb'))
    map_location = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    state = torch.load(state_path, map_location=map_location)

    model = config()
    model.to(device)
    model.load_state_dict(state, strict=True)

    logger.debug(f'\n[ utils/serialization ] Loaded config from {config_path}\n', main_process_only=True)
    logger.debug(config, main_process_only=True)
    return model, epoch

def load_optimizer(logger, optimizer, *loadpath, epoch=None, device='cuda:0', debug=False, type='prior'):
    loadpath = os.path.join(*loadpath)
    prefix = f"{type}_" if type != "" else ""

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath, f"{prefix}optimizer_", debug)
    
    if epoch < 0:
        return optimizer, epoch

    logger.debug(f'[ utils/serialization ] Loading optimizer epoch: {epoch}', main_process_only=True)
    state_path = os.path.join(loadpath, f'{prefix}optimizer_state_{epoch}.pt')

    map_location = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    state = torch.load(state_path, map_location=map_location)

    optimizer.load_state_dict(state)

    return optimizer, epoch

def load_scaler(logger, scaler, *loadpath, epoch=None, device='cuda:0', debug=False, type='prior'):
    loadpath = os.path.join(*loadpath)
    prefix = f"{type}_" if type != "" else ""

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath, f"{prefix}scaler_", debug)
    
    if epoch < 0:
        return scaler, epoch

    logger.debug(f'[ utils/serialization ] Loading optimizer epoch: {epoch}', main_process_only=True)
    state_path = os.path.join(loadpath, f'{prefix}scaler_state_{epoch}.pt')

    map_location = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    state = torch.load(state_path, map_location=map_location)

    scaler.load_state_dict(state)

    return scaler, epoch

def load_model(logger, *loadpath, epoch=None, device='cuda:0', data_parallel=False, debug=False):
    loadpath = os.path.join(*loadpath)
    config_path = os.path.join(loadpath, 'model_config.pkl')
    debug_suffix = '' if not debug else '_debug'

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath, debug=debug)

    logger.debug(f'[ utils/serialization ] Loading model epoch: {epoch}', main_process_only=True)
    state_path = os.path.join(loadpath, f'state_{epoch}{debug_suffix}.pt')

    config = pickle.load(open(config_path, 'rb'))
    map_location = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    state = torch.load(state_path, map_location=map_location)

    model = config()
    model.to(device)
    model.load_state_dict(state, strict=True)
    if data_parallel:
        # use all available cuda devices for data parallelization
        num_gpus = torch.cuda.device_count()
        model = torch.nn.DataParallel(model, device_ids=range(num_gpus))

    logger.debug(f'\n[ utils/serialization ] Loaded config from {config_path}\n', main_process_only=True)
    logger.debug(config, main_process_only=True)

    return model, epoch

def load_config(logger, *loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    logger.debug(f'[ utils/serialization ] Loaded config from {loadpath}', main_process_only=True)
    logger.debug(config, main_process_only=True)
    return config

def load_from_config(logger, *loadpath, **kwargs):
    config = load_config(logger, *loadpath)
    return config.make(**kwargs)

def load_args(*loadpath):
    from .setup import Parser
    loadpath = os.path.join(*loadpath)
    args_path = os.path.join(loadpath, 'args.json')
    args = Parser()
    args.load(args_path)
    return args
