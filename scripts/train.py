import os
import numpy as np
import torch
import json
import pdb
import pickle

import trajectory.utils as utils
from trajectory.utils.dataset import create_dataset
from trajectory.models.vqvae import VQContinuousVAE
from trajectory.utils.serialization import get_latest_epoch
import wandb
from accelerate.logging import get_logger

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.vqvae'

def save_model(args, trainer, model, save_epoch):
    ## save state, optimizer, scaler to disk
    tmp_model_statepath = os.path.join(args.savepath, f'tmp_state_{save_epoch}.pt')
    model_statepath = os.path.join(args.savepath, f'state_{save_epoch}.pt')
    model_state = trainer.accelerator.unwrap_model(model).state_dict()
    tmp_optimizer_statepath = os.path.join(args.savepath, f'tmp_optimizer_state_{save_epoch}.pt')
    optimizer_statepath = os.path.join(args.savepath, f'optimizer_state_{save_epoch}.pt')
    optimizer_state = trainer.accelerator.unwrap_model(trainer.optimizer).state_dict()

    trainer.accelerator.save(model_state, tmp_model_statepath)
    trainer.accelerator.save(optimizer_state, tmp_optimizer_statepath)
    

    ## rename saved tmp files to files
    ## save stats and dataloader
    if trainer.accelerator.is_main_process:
        os.rename(tmp_model_statepath, model_statepath)
        trainer.accelerator.print(f"Saved model to {model_statepath}\n")
        os.rename(tmp_optimizer_statepath, optimizer_statepath)
        trainer.accelerator.print(f"Saved optimizer to {optimizer_statepath}\n")
        statspath = os.path.join(args.savepath, 'stats.pkl')
        with open(statspath, 'wb') as f:
            pickle.dump(trainer.stats, f)
            trainer.accelerator.print(f"Saved stats: {trainer.stats}\n")
        trainer.loader.save()
        trainer.accelerator.print(f"Saved dataloder state\n")
    accelerator_statepath = os.path.join(args.savepath, f'accelerator_state')
    trainer.accelerator.save_state(output_dir=accelerator_statepath)
    trainer.accelerator.print(f"Saved accelerator state to {accelerator_statepath}\n")

def main():

    args = Parser().parse_args('train')
    args.n_layer = int(args.n_layer)

    logger = get_logger(__name__, log_level="DEBUG")

    #######################
    ####### loading #######
    #######################

    args.logbase = os.path.expanduser(args.logbase)
    args.savepath = os.path.expanduser(args.savepath)
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    
    #######################
    ####### trainer #######
    #######################

    # use all available cuda devices for data parallelization
    num_gpus = torch.cuda.device_count()
    logger.debug(f"Using {num_gpus} gpus.\n", main_process_only=True)

    n_tokens_target = int(args.n_tokens_target)
    n_epochs_ref = int(args.n_epochs_ref)
    warmup_tokens = n_epochs_ref * n_tokens_target * 0.2
    final_tokens = n_epochs_ref * n_tokens_target

    trainer_config = utils.Config(
        utils.VQTrainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        # optimization parameters
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        grad_norm_clip=1.0,
        weight_decay=0.1, # only applied on matmul weights
        # learning rate decay: linear warmup followed by cosine decay to 10% of original
        lr_decay=False,
        warmup_tokens=warmup_tokens,
        kl_warmup_tokens=warmup_tokens*10,
        final_tokens=final_tokens,
        ## dataloader
        num_workers=int(args.num_workers),
        device=args.device,
        train_batch_size=int(args.train_batch_size),
        load_batch_size=int(args.load_batch_size),
        n_tokens_target=n_tokens_target,
        enable_fp16=args.enable_fp16,
    )

    trainer = trainer_config()

    ############################
    ######## DataLoader ########
    ############################
    
    dataset = create_dataset(args)

    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = obs_dim + act_dim + 3

    #######################
    ######## model ########
    #######################

    block_size = args.subsampled_sequence_length * transition_dim # total number of dimensionalities for a maximum length sequence (T)

    logger.debug(
        f'Joined dim: {transition_dim} '
        f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}', 
        main_process_only=True)


    model_config = utils.Config(
        VQContinuousVAE,
        savepath=(args.savepath, 'model_config.pkl'),
        ## discretization
        vocab_size=args.N, block_size=block_size,
        K=args.K,
        code_per_step=args.code_per_step,
        ## architecture
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd * args.n_head,
        ## dimensions
        observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
        ## loss weighting
        action_weight=args.action_weight, reward_weight=args.reward_weight, value_weight=args.value_weight,
        position_weight=args.position_weight,
        trajectory_embd=args.trajectory_embd,
        model=args.model,
        latent_step=args.latent_step,
        ma_update=args.ma_update,
        residual=args.residual,
        obs_shape=args.obs_shape,
        ## dropout probabilities
        embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
        bottleneck=args.bottleneck,
        masking=args.masking,
        ae_type=args.ae_type,
        state_conditional=args.state_conditional,
        use_discriminator=args.use_discriminator,
        disc_start=args.disc_start,
        blocks_per_layer=args.blocks_per_layer,
        encoder_inputs=args.encoder_inputs,
        position_embedding=args.position_embedding,
        causal_attention=args.causal_attention,
        causal_conv=args.causal_conv,
        symlog=args.symlog,
        data_parallel=args.data_parallel,
    )
    model = model_config()

    # initialize or load stats
    statspath = os.path.join(args.savepath, 'stats.pkl')
    if os.path.isfile(statspath):
        with open(statspath, 'rb') as f:
            stats = pickle.load(f)
            logger.debug(f"Loaded stats: {stats}\n", main_process_only=True)
        model, _ = utils.load_model(logger, args.logbase, args.dataset, args.exp_name, epoch="latest", device=args.device)
    else:
        stats = {
            "n_epochs": 0,
            "n_tokens": 0, # counter used for learning rate decay
            "last_save_n_tokens": 0,
            "last_logging_n_tokens": 0,
            "n_steps": 0,
            "n_logging": 0,
        }
        trainer.accelerator.print("Training from scratch...\n")

    model.set_padding_vector(np.zeros(model.transition_dim-1))

    #######################
    ###### main loop ######
    #######################

    ## scale number of epochs to keep number of updates constant
    n_saves = int(args.n_saves)
    save_freq = int(n_epochs_ref // n_saves)
    wandb_conf = {"entity": "transferplan", "group": args.exp_name, "reinit": True, "config": args, "tags": [args.exp_name, args.tag]}
    trainer.init_stats(stats)
    trainer.init_data_loader(dataset, stats)
    trainer.init_wandb(wandb_conf, name=args.exp_name)
    trainer.accelerator.print(f'model parameters {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000}M')

    # load accelerator state
    accelerator_statepath = os.path.join(args.savepath, f'accelerator_state.pt')
    if os.path.exists(accelerator_statepath):
        trainer.accelerator.load_state(accelerator_statepath)

    optimizer = trainer.get_optimizer(model)
   
    # load optimizer and scaler if needed
    optimizer, _ = utils.load_optimizer(logger, optimizer, args.logbase, args.dataset, args.exp_name,
                                            epoch="latest", device=args.device, type="")

    model, trainer.optimizer = trainer.accelerator.prepare(model, optimizer)

    start_ep = trainer.stats['n_epochs']
    for epoch in range(start_ep, n_epochs_ref):
        trainer.accelerator.print(f'\nEpoch: {epoch} / {n_epochs_ref} | {args.dataset} | {args.exp_name}')

        trainer.train(model, dataset, save_freq=1e4, savepath=args.savepath)

        if epoch % 10 == 0:
            ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
            save_epoch = (epoch + 1) // save_freq * save_freq
            save_model(args, trainer, model, save_epoch)
    
    # save the final trained model
    save_epoch = n_epochs_ref // save_freq * save_freq
    save_model(args, trainer, model, save_epoch)
            
if __name__ == '__main__':
    main()
