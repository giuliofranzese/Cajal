import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from setproctitle import setproctitle

import torch
import numpy as np
import random
import scipy
from tqdm import tqdm
from scipy.special import lambertw # for importance sampling
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from pprint import pprint
import yaml
import argparse
import logging
import tempfile
from pathlib import Path
import pytorch_lightning as pl
import math
from torch.utils.data import TensorDataset, DataLoader, random_split
import re
 
import argparse
import logging
import yaml

import os
import sys
sys.path.append('../')
 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")# Set default GPU for data generation (can be overridden by individual scripts)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from functional_fm import *
from models.models_fno.fno import FNO 
from models.models_transformer.Transformer_TS import pmfDiT
from models.models_mino.minot_ts import MINOT_TS
from util.ema import EMA
from util.util import load_checkpoint, add_config_to_argparser, plot_spectrum_comparison, seed_everything, apply_overrides
from util.combined_optimizer import build_muon_optimizer


 
from functional_kl import *
 
from kl_functions_torch import noise_x0_class
basis = 'fourier' 






################################################################################
# Create data
################################################################################
def getX0X1(data_params, n_samples, upsample,
            ):

    global basis
    
    M = data_params['general']['M']
    D = data_params['general']['D']
    
    x0source = data_params['general']['x0source']
    amp_way = data_params['general']['amp_way']
   

    white_noise = data_params['X0_GP_matern']['white_noise']

 
    x1_org   = torch.from_numpy(
        np.load(
            data_params['X1']['A_path'][int(M*upsample)]
        )).float().to( device=device)
    
    x1_org_B = torch.from_numpy(np.load(data_params['X1']['B_path'][int(M*upsample)])).float().to( device=device)


    # x1_org   = x1_org[  :, :int(M*upsample), :]
    # x1_org_B = x1_org_B[:, :int(M*upsample), :]
    print(x1_org.shape)
 
    # mu = x1_org.mean(dim=-1) # dim=0, keepdim=True)
    mu      = x1_org.mean(dim=[0,1], keepdim=True) # [1, 1, 3])
    std_x1A = x1_org.std(dim=[0,1],  keepdim=True) # [1, 1, 3]) 


 
    x1_time_mean0   = ( x1_org    - mu )  / std_x1A   
    x1_time_mean0_B = ( x1_org_B  - mu )  / std_x1A
    print(f'A : mean={x1_time_mean0.mean()}, std={x1_time_mean0.std()}' )
    print(f'B : mean={x1_time_mean0_B.mean()}, std={x1_time_mean0.std()}' )

    x1_time_mean0_forx0 = x1_time_mean0
   


   
    # ########################## #### changed with B for sde case 
    x1_org_forx0 = torch.from_numpy(np.load( 
            data_params['X1']['A_path'][int(M*upsample)]#.replace('gt', 
                                                        #         x0source, # 'am'
                                                        #         ) 
        )).float().to( device=device)
    x1_time_forx0_mean0   = ( x1_org_forx0    - mu )  / std_x1A  
    x1_time_mean0_forx0 = x1_time_forx0_mean0
    # amp_way   = None
    # ########################## 


    # Noise type
    noise_type = data_params['X0_GP_matern'].get('noise_type', 'data')

    if noise_type == 'matern':
        # Matérn kernel noise — no data covariance needed
        matern_params = {
            'nu_K':    data_params['X0_GP_matern']['nu_K'],
            'ell_K':   data_params['X0_GP_matern']['ell_K'],
            'sigma2_K': data_params['X0_GP_matern']['sigma2_K'],
            'M':       int(M * upsample),
        }
        sample_X0_data_obj = noise_x0_class(
            x1_time_mean0=x1_time_mean0_forx0,
            noise_type='matern',
            matern_params=matern_params,
        )
    elif noise_type == 'empirical_decay':
        noise_params = {
            'n_empirical': data_params['X0_GP_matern'].get('n_empirical', 32),
            'eps_decay':   data_params['X0_GP_matern'].get('eps_decay', 0.1),
        }
        sample_X0_data_obj = noise_x0_class(
            x1_time_mean0=x1_time_mean0_forx0,
            noise_type='empirical_decay',
            noise_params=noise_params,
            amp_way=amp_way,
        )
    else:
        # Data-driven noise (white / cov / rough)
        sample_X0_data_obj = noise_x0_class(
            x1_time_mean0=x1_time_mean0_forx0,
            noise_amp=None,
            white_noise=white_noise,
            amp_way=amp_way,
        )
    # sample_X0_data_obj = noise_x0_class( x1_time_mean0=x1_time_mean0_forx0, noise_amp=None, white_noise=white_noise,
    #                                      amp_way = amp_way
    #                                    )
    
   
    _ = sample_X0_data_obj.sample(n_samples=n_samples, upsample=upsample,
                        return_hat = True
                        )
  
   
    lam_k_K = sample_X0_data_obj.lam_k
    Phi = sample_X0_data_obj.Phi

    x1_org = np.load(data_params['X1']['A_path'][int(M*upsample)])

    data_dict = {
        'A': {'X1_data': x1_time_mean0[:n_samples]     
            },
        'B': {'X1_data': x1_time_mean0_B[:n_samples]  
            }
    }
    # # to torch
    # x1_org   = torch.from_numpy(x1_org).float().to( device=device)
    
    # data_dict = {
    #     'A': {'X1_data': x1_org[:n_samples]     
    #         },
    #     'B': {'X1_data': x1_org_B[:n_samples]  
    #         }
    # }

    return data_dict,      sample_X0_data_obj, lam_k_K, Phi


    



################################################################################
# Create ffm model
################################################################################
def create_ffm(
        # Phi, lam_k_C, lam_k_K, 
        sample_X0_data_obj,
        t_train_sampling_scheme, prediction, loss_lam_time,
        D, M,

        modes,
        width,
        mlp_width,
        x_dim,

        loss_decay=False,

        **kwargs,

        ):

    model_type = kwargs.pop('model_type', 'transformer')
    mino_params = kwargs.pop('mino_params', {})

    # decide what model to use 
    if model_type == 'mino_t': # MINO implementation 
        model = MINOT_TS(
            input_size=M,
            in_channels=D,
            **mino_params,
        )
    else: # tranformer implementation
        model = pmfDiT( 
            input_type = 'ts',
            input_size  = M,

            in_channels = D,

            patch_size  = 2,
            hidden_size = 256,
            depth = 2, # 5, # 16,
        )

    model = model.to(device)
    
    fmot = FFMModel(model=model,
                    D=D,
                    device=device,
                    sample_X0_data_func = sample_X0_data_obj,

                    x_dim=x_dim,

                    prediction=prediction,
                    t_train_sampling_scheme=t_train_sampling_scheme,

                    loss_lam_time=loss_lam_time,
                    loss_decay=loss_decay,

                    curriculum_sampling=kwargs.pop('curriculum_sampling', False),
                    curriculum_switch_frac=kwargs.pop('curriculum_switch_frac', 0.4),
                    curriculum_logit_mean=kwargs.pop('curriculum_logit_mean', 0.8),
                    curriculum_logit_std=kwargs.pop('curriculum_logit_std', 1.0),
                   )
    
    return fmot



def trainer(data_params, nn_params,    
   upsample_gen,


  n_samples_train,
  n_samples_gen, 
 
  t_train_sampling_scheme,
  prediction,
  loss_lam_time,

  batch_size,
  num_iterations,

   
  ema_opti,


  sfolder,


  epochs_selected=-1,

  curriculum_sampling=False,
  curriculum_switch_frac=0.4,
  curriculum_logit_mean=0.8,
  curriculum_logit_std=1.0,

  lr=1e-4,
  lr_sch_step =  50,
  lr_gamma = 0.1,
  loss_decay=False,
  sweep_mode=False,
  opt_name="adam",

         ):
    
   

    #########################################################################################
    # Data generation phase -- for Train
    data_dict,      sample_X0_data_obj, lam_k_K, Phi = getX0X1(
        data_params=data_params, 

        n_samples=n_samples_train, 
        upsample=1, 
      
    )
    # Done
    ###########################################################################################


    ################################################################################
    # Train or Generate
    ################################################################################
    results = {} 
    for y, id in enumerate(['A', 'B']): 
        X1_data = data_dict[id]['X1_data']
        results[id] = { 
            'x_train': X1_data, 
            'y_train': ( torch.ones(X1_data.shape[0]) * y ).long()
        } 

 
    
    # data
    dataset_tr = TensorDataset(
        torch.cat([
                    results['A']['x_train'],
                    results['B']['x_train'],
                ]),
        torch.cat([
                    results['A']['y_train'],
                    results['B']['y_train'],
                ]),
        )  
    loader_tr  = DataLoader(
        dataset_tr,
        batch_size=batch_size,
        shuffle=True
    )

    n_samples_train_real = len(dataset_tr)
    steps_per_epoch = math.ceil(n_samples_train_real / batch_size) 
    epochs = math.ceil(num_iterations / steps_per_epoch)  # 1000 # 300 
    print(f'training epochs: {epochs}, steps_per_epoch: {steps_per_epoch}, batch_size: {batch_size}, n_samples_train={n_samples_train_real}') 


    # model
    fmot = create_ffm(
        sample_X0_data_obj,
            t_train_sampling_scheme, prediction, loss_lam_time,
            **data_params['general'],
            **nn_params,
            loss_decay=loss_decay,
            curriculum_sampling=curriculum_sampling,
            curriculum_switch_frac=curriculum_switch_frac,
            curriculum_logit_mean=curriculum_logit_mean,
            curriculum_logit_std=curriculum_logit_std,
            )

    ## save path
    spath = Path(f'../{sfolder}')
    spath.mkdir(parents=True, exist_ok=True)

    os.makedirs(spath / 'imgs', exist_ok=True)
    os.makedirs(spath / 'ckpt', exist_ok=True)

    epochs_selected = epochs_selected if epochs_selected != -1 else epochs   
    epoch_path = Path(spath / f'ckpt/epoch_{epochs_selected}.pt' )
    print(epoch_path) 


    if not epoch_path.exists(): 
        print(f'to train')

        if not sweep_mode:
            wandb_run = wandb.init(
                project='fkl_flag',
                name=str(sfolder).split("/")[-1],
                reinit=True,
            )
        

        # Optimizer 
        if opt_name == "muon":
            optimizer = build_muon_optimizer(fmot.model, 
                                                
                lr=lr ,
            optimizer_weight_decay=1e-3, 
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                fmot.model.parameters(), lr=lr, fused=True
            )
        if ema_opti:
            optimizer = EMA(optimizer, ema_decay=0.999)
            

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0000003
        )


        # train!
        fmot.train(loader_tr, 
                    optimizer=optimizer, 
                    epochs=epochs, 
                    scheduler=scheduler, 

                eval_int= 200, 
                generate=n_samples_gen,  

                save_path=spath,  

                test_loader = None
                
                ) 

        if not sweep_mode:
            wandb.finish()
 
 

    # else:
    fmot.model = load_checkpoint(epoch_path, fmot.model)[0]
    fmot.model.eval() 
    print('load saved ckpt --- done!') 

    for upsample in [1]: # list((upsample_gen)): 
        pdf_path = spath / 'imgs' /  f'epoch_{epochs_selected}_gen_upsample{upsample}_A.pdf'
        if   not pdf_path.exists():   
            print(f'to gen, upsample = {upsample}')


            batch_xA = loader_tr.dataset[:n_samples_gen][0]
            batch_xB = loader_tr.dataset[-n_samples_gen:][0]

            # samples = self.sample(n_samples=generate)
            samples, x0 = fmot.sample(  
                        n_samples= n_samples_gen, # 5_000 if  upsample != 16 else 1_000 , #
                        upsample = 1,
                        return_x0=True) # [n_gen_samples, n_channels, n_x * upsample]

            plot_real_vs_fake(
                        y_real=batch_xA, 
                        y_fake=samples[0], 
                        save_path=spath / 'imgs' /  f'epoch_{epochs_selected}_gen_upsample{upsample}_A.pdf',  
                        )
            plot_real_vs_fake(
                        y_real=batch_xB, 
                        y_fake=samples[1], 
                        save_path=spath / 'imgs' /  f'epoch_{epochs_selected}_gen_upsample{upsample}_B.pdf',  
                        )

            # breakpoint()
            
            plot_spectrum_comparison(x1_time=batch_xA, x1_time_gen=samples[0], x0_time=x0, 
                        save_path=spath / 'imgs' /  f'epoch_{epochs_selected}_gen_upsample{upsample}_A_energyspectrum.pdf', 
                        )
            plot_spectrum_comparison(x1_time=batch_xB, x1_time_gen=samples[1], x0_time=x0, 
                        save_path=spath / 'imgs' /  f'epoch_{epochs_selected}_gen_upsample{upsample}_B_energyspectrum.pdf',  
                        )

    
    for y, id in enumerate(['A', 'B']): 
        results[id].update({
            'fmot': fmot,
            'epoch_path': epoch_path,
        })



    return results 


 

def kl_estimate(data_params, 
                
                results,
                sfolder,
            
                n_t,  
                t_kl_sampling_scheme,

                n_samples_kl,

                upsample_kl,


                vdiff_source,
                v_source,

           
                 
                # **kwargs
                ):   
    
    ################################################################################
    # Compute KL
    ################################################################################

    KL_est_mean_dict = {}
    for upsample in list((upsample_kl)): # set  [1] + 
        KL_est_mean_dict[upsample] = {}
    
        print(f'to estimate kl, upsample = {upsample}')


        ###########################################################################################
        # Data generation phase -- for KL
        data_dict,      sample_X0_data_obj, lam_k_K, Phi = getX0X1(
            data_params=data_params, 

            n_samples=n_samples_kl, 
            upsample=upsample,
        )
        X1_data_A = data_dict['A']['X1_data']
        # Done
        ###########################################################################################
 
        for KL_direction, X1_data_A_KL in  { 
            'forward_KL': data_dict['A']['X1_data'],
            'reverse_KL': data_dict['B']['X1_data'] 
                              }.items() : 
            ###########################################################################################
            # KL estimation phase

            functional_kl_obj = FKLModel( 
                ffm_model_A = results['B']['fmot'] , # they are the same model so we can assign the same
                ffm_model_B = results['B']['fmot'] ,
            )

            
            print('kl_estimate_func: kl_estimate_gen')
            KL_est_mean, KL_est_std, KL_est = functional_kl_obj.kl_estimate_noisediffchannel( 
                        X1_data_A=X1_data_A_KL,  
                        
                        # ---- stats ----
                        upsample=upsample, 
                        Phi=Phi, 
                        # analytic_stats = [mean_hat_A, mean_hat_B, lam_k_C],
                        lam_k_K=lam_k_K, 
                                            
                        # ---- Integration over FFM's t ----
                        n_t = n_t, 
                        t_kl_sampling_scheme = t_kl_sampling_scheme,  

                        # ---- (diff of) v_t 's source ----
                        vdiff_source = vdiff_source, 
                        v_source     = v_source  ,

                        # ---- save check mse error plot ----
                        sfolder = sfolder,
                        )

            if vdiff_source == 'sample':
                print(f"sample-wise KL: {KL_est_mean:.3f} ± {KL_est_std:.3f}")

          
            KL_est_mean_dict[upsample][KL_direction] =  KL_est_mean
    

    return KL_est_mean_dict
        

 

def main(
      config,
      sweep_mode=False,
        ):

    if sweep_mode:
        config['trainer_params']['sweep_mode'] = True
    sfolder = config.get("trainer_params")['sfolder']

    results = trainer(
        data_params=config.get("data_params"), 
        nn_params=config.get("nn_params") ,
 
        **config.get("trainer_params"),    
         )
    
    KL_est_mean_dict = kl_estimate(data_params=config.get("data_params"), 
                results=results, 
                sfolder=sfolder,
                **config.get("kl_params"),  
    )

    config['kl_result'] = KL_est_mean_dict


    # if config.get("kl_params")['t_kl_sampling_scheme'] == 'uniform':
    #     save_file = f"../{sfolder}/config_kl_FINAL_klt_uniform.yaml" # trainer_params.epochs_selected
    # else:
    #     save_file = f"../{sfolder}/config_kl_FINAL.yaml"
    
 
    save_file = f"../{sfolder}/config_kl_FINAL"
    save_file += ( "_" + str(config.get("trainer_params")['epochs_selected']) )
    if config.get("kl_params")['t_kl_sampling_scheme'] == 'uniform':
        save_file += "_klt_uniform"
    save_file += ".yaml"


    with open(save_file , "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,     
            default_flow_style=False
        )
    print('result saved to ',  save_file)






 
if __name__ == "__main__":

    

    setproctitle("main")

    seed_everything(42)
 
  
    # Setup Logger
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,  help="Path to yaml config")
    
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values from CLI, e.g. --set train.lr=1e-4 --set model.hidden=128",
    )
    parser.add_argument("--sweep", action="store_true", help="Run as wandb sweep agent")


    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    config = apply_overrides(config, args.overrides)

    print(yaml.dump(config, sort_keys=False))
 
 
    # if not config['data_params']['X0_GP_matern']['white_noise']:
    #     config['trainer_params']['sfolder'] += f"_X0rougher"
    # else:
    #     config['trainer_params']['sfolder'] += '_X0white'
    # config['trainer_params']['sfolder'] += "_losst"+str(config['trainer_params']['loss_lam_time'])
    # if config['trainer_params']['t_train_sampling_scheme'] == "importance_sampling_t/(1-t)" :
    #     config['trainer_params']['sfolder'] += "_traintIStail"
    # elif config['trainer_params']['t_train_sampling_scheme'] == "importance_sampling_t*(1-t)" :
    #     config['trainer_params']['sfolder'] += "_traintISmid"
    # elif config['trainer_params']['t_train_sampling_scheme'] == "uniform" :
    #     config['trainer_params']['sfolder'] += "_traintUniform"
    # elif config['trainer_params']['t_train_sampling_scheme'] == "logit_normal":
    #     config['trainer_params']['sfolder'] += "_traintLogit_normal"

 

    spath = Path(f"../{config['trainer_params']['sfolder']}")
    

    print(spath)
    # breakpoint()
    spath.mkdir(parents=True, exist_ok=True)



    save_file = spath / "config_pretrain.yaml"
    if not save_file.exists(): 
        with open(save_file , "w") as f:
            yaml.safe_dump(
                config,
                f,
                sort_keys=False,      # keep your key order
                default_flow_style=False
            )
        print('config_pretrain saved to ',  save_file)


    if args.sweep:
        def sweep_train():
            wandb.init()  # sweep agent initializes the run

            # Override from wandb sweep config
            sweep_cfg = dict(wandb.config)

            # Separate MINO arch params from trainer params
            mino_override_keys = {'enc_dim', 'enc_depth', 'dec_depth', 'enc_num_heads', 'radius'}
            for key in list(sweep_cfg.keys()):
                if key in mino_override_keys:
                    val = sweep_cfg.pop(key)
                    config['nn_params']['mino_params'][key] = val
                    # Tie enc/dec dims and heads
                    if key == 'enc_dim':
                        config['nn_params']['mino_params']['dec_dim'] = val
                    if key == 'enc_num_heads':
                        config['nn_params']['mino_params']['dec_num_heads'] = val

            # Override trainer_params with remaining sweep params
            for key, val in sweep_cfg.items():
                config['trainer_params'][key] = val

            # Set unique sfolder per run
            config['trainer_params']['sfolder'] = f"log/GM/sweep/{wandb.run.id}"

            # Recreate save dir
            spath_sweep = Path(f"../{config['trainer_params']['sfolder']}")
            spath_sweep.mkdir(parents=True, exist_ok=True)
            save_file_sweep = spath_sweep / "config_pretrain.yaml"
            if not save_file_sweep.exists():
                with open(save_file_sweep, "w") as f:
                    yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

            main(config, sweep_mode=True)

        sweep_train()
    else:
        main(config)