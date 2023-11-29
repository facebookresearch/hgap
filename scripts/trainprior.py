import os
import numpy as np
import torch
import pickle
import gc
from GPUtil import showUtilization as gpu_usage

import trajectory.utils as utils
from trajectory.utils.dataset import create_dataset
from trajectory.models.transformer_prior import TransformerPrior
from accelerate.logging import get_logger

os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
torch.backends.cuda.matmul.allow_tf32 = True

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.vqvae'

def save_model(args, trainer, model, save_epoch):
    ## save state, optimizer to disk
    tmp_model_statepath = os.path.join(args.savepath, f'tmp_{args.type}_state_{save_epoch}.pt')
    model_statepath = os.path.join(args.savepath, f'{args.type}_state_{save_epoch}.pt')
    model_state = trainer.accelerator.unwrap_model(model).state_dict()
    tmp_optimizer_statepath = os.path.join(args.savepath, f'tmp_{args.type}_optimizer_state_{save_epoch}.pt')
    optimizer_statepath = os.path.join(args.savepath, f'{args.type}_optimizer_state_{save_epoch}.pt')
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
        statspath = os.path.join(args.savepath, f'{args.type}_stats.pkl')
        with open(statspath, 'wb') as f:
            pickle.dump(trainer.stats, f)
            trainer.accelerator.print(f"Saved stats: {trainer.stats}\n")
        trainer.loader.save()
        trainer.accelerator.print(f"Saved dataloder state\n")
    accelerator_statepath = os.path.join(args.savepath, f'{args.type}_accelerator_state')
    trainer.accelerator.save_state(output_dir=accelerator_statepath)
    trainer.accelerator.print(f"Saved accelerator state to {accelerator_statepath}\n")

def main():
    gc.collect()
    torch.cuda.empty_cache()
    # print("Initial GPU Usage")
    # gpu_usage() 
    #######################
    ######## setup ########
    #######################
    logger = get_logger(__name__, log_level="DEBUG")

    args = Parser().parse_args('plan')
    if "vae_name" in args.__dict__ and args.vae_name != "":
        vae_name = args.vae_name
    else:
        vae_name = args.exp_name

    representation, _ = utils.load_model(logger, args.logbase, args.dataset, vae_name, epoch=args.gpt_epoch, device=args.device)
    args = Parser().parse_args('train')

    sequence_length = args.subsampled_sequence_length * args.step
    args.logbase = os.path.expanduser(args.logbase)
    args.savepath = os.path.expanduser(args.savepath)
    args.code_per_step = int(args.code_per_step)
   
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
     ## HACK: to avoid launch duplicate job
    if os.path.exists(os.path.join(args.savepath, 'critic_state_600.pt')):
        return

    dataset = create_dataset(args)

    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim

    transition_dim = obs_dim + act_dim + 3

    representation.set_padding_vector(np.zeros(representation.transition_dim - 1))

    obs_dim = dataset.observation_dim

    model_config = utils.Config(
        TransformerPrior,
        savepath=(args.savepath, f'{args.type}_model_config.pkl'),
        ## discretization
        K=representation.K, max_sequence_length=args.subsampled_sequence_length,
        ## architecture
        observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
        n_layer=args.prior_layer, n_head=args.prior_head, n_embd=args.prior_embd * args.prior_head,
        value_layer=args.value_layer, value_head=args.value_head, value_embd=args.value_embd * args.value_head,
        ## loss weighting
        latent_step=args.latent_step,
        code_per_step=representation.code_per_step,
        ## dropout probabilities
        embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
        obs_shape=args.obs_shape,
        position_embedding=args.position_embedding,
        latent_steps=representation.latent_step,
        data_parallel=args.data_parallel,
        twohot_value=args.twohot_value,
        value_ema_rate=args.value_ema_rate,
        tau=args.tau,
        cql_weight=args.cql_weight,
    )
    

    num_gpus = torch.cuda.device_count()
    logger.debug(f"Using {num_gpus} gpus.\n", main_process_only=True)

    #######################
    ####### trainer #######
    #######################

    n_tokens_target = int(args.n_tokens_target)
    n_epochs_ref = int(args.n_epochs_ref)
    warmup_tokens = n_epochs_ref*n_tokens_target*0.2
    final_tokens = n_epochs_ref*n_tokens_target

    trainer_config = utils.Config(
        utils.PriorTrainer,
        savepath=(args.savepath, f'{args.type}trainer_config.pkl'),
        # optimization parameters
        train_batch_size=int(args.train_batch_size),
        load_batch_size=int(args.load_batch_size),
        learning_rate=args.prior_learning_rate,
        betas=(0.9, 0.95),
        grad_norm_clip=2.0 if "prior_gradient_norm_clip" not in args.__dict__ else args.prior_gradient_norm_clip,
        weight_decay=0.1, # only applied on matmul weights
        # learning rate decay: linear warmup followed by cosine decay to 10% of original
        lr_decay=args.lr_decay,
        warmup_tokens=warmup_tokens,
        kl_warmup_tokens=warmup_tokens*10,
        final_tokens=final_tokens,
        ## dataloader
        num_workers=args.num_workers,
        device=args.device,
        n_tokens_target=n_tokens_target,
        discount=args.discount,
        enable_fp16=args.enable_prior_fp16,
        bootstrap=args.bootstrap,
        bootstrap_ignore_terminal=args.bootstrap_ignore_terminal,
        type=args.type,
    )

    trainer = trainer_config()

    
    # initialize or load stats
    statspath = os.path.join(args.savepath, f'{args.type}_stats.pkl')
    if os.path.isfile(statspath):
        with open(statspath, 'rb') as f:
            stats = pickle.load(f)
            logger.debug(f"Loaded stats: {stats}\n", main_process_only=True)
        # if trainer.accelerator.is_main_process:
        model, _ = utils.load_transformer_model(logger, args.logbase, args.dataset,
                                                args.prior_name if args.type=="prior_finetune" else args.exp_name,
                                                epoch="latest", device=args.device, type=args.type)
        dataset.restore()
    elif args.type == "prior_finetune":
        model, _ = utils.load_transformer_model(logger, args.logbase, args.dataset,
                                                args.prior_name if args.type=="prior_finetune" else args.exp_name,
                                                epoch="latest", device=args.device, type="prior")
        dataset.restore()
        stats = {
            "n_epochs": 0,
            "n_tokens": 0,  # counter used for learning rate decay
            "last_save_n_tokens": 0,
            "last_logging_n_tokens": 0,
            "n_steps": 0,
            "ema_reconstruction": 0,
            "n_logging": 0,
        }
    else:
        model = model_config()
        stats = {
            "n_epochs": 0,
            "n_tokens": 0, # counter used for learning rate decay
            "last_save_n_tokens": 0,
            "last_logging_n_tokens": 0,
            "n_steps": 0,
            "ema_reconstruction": 0,
            "n_logging": 0,
        }
    if args.type == "critic" and args.prior_name != "":
        if trainer.accelerator.is_main_process:
            prior_model, _ = utils.load_transformer_model(logger, args.logbase, args.dataset, args.prior_name,
                                                        epoch="latest", device=args.device, type="prior")
    else:
        prior_model = None

    if args.type == "prior_finetune":
        if trainer.accelerator.is_main_process:
            critic_model, _ = utils.load_transformer_model(logger, args.logbase, args.dataset, args.critic_name,
                                                           epoch="latest", device=args.device, type="critic")
            critic_model.eval()
    else:
        critic_model = None

    # print("After model init GPU Usage")
    # gpu_usage() 

    #######################
    ###### main loop ######
    #######################

    ## scale number of epochs to keep number of updates constant
    n_saves = int(args.n_saves)
    save_freq = int(n_epochs_ref // n_saves)
    # wandb.init(project="latentPlanning", entity="transferplan", group=args.exp_name, reinit=True, config=args, tags=[args.exp_name, args.tag, "prior"])
    wandb_conf = {"entity": "transferplan", "group": args.exp_name, "reinit": True, "config": args, "tags": [args.exp_name, args.tag, args.type]}
    trainer.init_stats(stats)
    trainer.init_data_loader(dataset)
    trainer.init_wandb(wandb_conf, name=args.exp_name)
    trainer.accelerator.print(f'model parameters {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000}M')

    optimizer = trainer.get_optimizer(model)

    # load optimizer if needed
    if trainer.accelerator.is_main_process:
        optimizer, _ = utils.load_optimizer(logger, optimizer, args.logbase, args.dataset, args.exp_name,
                                                epoch="latest", device=args.device, type=args.type)

    torch.cuda.empty_cache()
    model, trainer.optimizer = trainer.accelerator.prepare(model, optimizer)
    # load accelerator state
    accelerator_statepath = os.path.join(args.savepath, f'{args.type}_accelerator_state.pt')
    if os.path.exists(accelerator_statepath):
        trainer.accelerator.load_state(accelerator_statepath)
    if prior_model:
        prior_model = trainer.accelerator.prepare(prior_model)
    if prior_model:
        critic_model = trainer.accelerator.prepare(critic_model)

    start_ep = trainer.stats['n_epochs']
    for epoch in range(start_ep, n_epochs_ref):
        trainer.accelerator.print(f'\nEpoch: {epoch} / {n_epochs_ref} | {args.dataset} | {args.exp_name}')

        nan_loss = trainer.train(representation, model, dataset, type=args.type, prior_model=prior_model, critic_model=critic_model)
        if nan_loss:
            trainer.accelerator.print(f"Training aborted due to NaN losses!\n")
            return

        ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
        if epoch % 10 == 0:
            save_epoch = (epoch + 1) // save_freq * save_freq
            save_model(args, trainer, model, save_epoch)
    
    # save the final trained model
    save_epoch = n_epochs_ref // save_freq * save_freq
    save_model(args, trainer, model, save_epoch)
    

if __name__ == '__main__':
    main()
