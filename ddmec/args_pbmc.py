import argparse
BS =64
def get_namespace():
    # Store default arguments in a dictionary
    default_args = {  
        "num_timesteps": 1000,
        "betas": [0.0001, 0.02],
        "num_steps_train": 50,
        "schedule": "linear",
        "guidance": 7,
        "mc_steps": 5,
        "mc_steps_z": 1,
        "kl_weight": [0.01, 0.01],
        "reward_weight": [1, 1],
        "intervals" : [-1,1],
        "bs": 128,
        "mlp":0,
        "batch_size_sample":128,
        "batch_size_train": 128*10, ## 50*128 /1600 = 4 updates
        "num_grad_accummulate": 5,
        "batch_size_update": 64, ## 128 + 256 /64  *2 = 12 updates
        "nb_update_pg": 1,
        "nb_update_reward": 1,
        "lr": [1e-4, 5e-4],
        "reward": "vlb",
        "warmup": 1,
        "max_step": 170000,
        "log_step": 500,
        "seed": 88,
        "ema": 0,
        "clip_norm": [1, 1],
        "advantage": 0,
        "uncond_p": [1.0 ,0.1],
        "grad_scale": [1.0,1.0],
        "clip_gen": 0,
        "diff": "ddpm",
        "method": "dpok",
        "reduce": "mean",
        "nb_device": 1,
        "pg_importance_sampling": 1,
        "train_uncond": 1,
        "one_step": 1,
        "neighbor":1,
        "lora": 0,
        "clip_range_pg": 0.0001,
        "eps": [0.02, 0.02],
        "log_nb_images": 4,
        "mod_name": "snare",      
        "path_log":"snare",  
        "transform":"minmax",
        "cond_method": "concat",
        "loss_type":"mse",
        "path": "",
        "reply_buffer":0,
        "size_buffer":256,
        "eta":1.0,
        "sampling":"random",
        "weighted":False,
    }

    # Instantiate the ArgumentParser
    parser = argparse.ArgumentParser(description="Training Arguments")
    
    # Dynamically add arguments from the default_args dictionary
    for arg, value in default_args.items():
        arg_type = type(value[0]) if isinstance(value, list) else type(value)
        nargs = "+" if isinstance(value, list) else None
        parser.add_argument(f"--{arg}", type=arg_type, default=value, nargs=nargs, help=f"Default: {value}")
    return parser