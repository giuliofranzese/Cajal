import torch
import pytorch_lightning as pl
from ..diffusion.sde import *
from ..diffusion.utils import EMA
from copy import deepcopy
from sklearn.metrics import mutual_info_score
from ..diffusion.ddpm_n import DDPM
from ..diffusion.iddpm_p2 import GaussianDiffusion

from ..discrete.utils import *
from sklearn.metrics import accuracy_score
import torch.distributed as dist
from ..eval.utils import save_image, calculate_ssim, calculate_psnr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import SCOT.src.evals as evals
import pandas as pd
import scanpy as sc
from scTopoGAN.Qualitative_Metrics import evaluate_neighbors

class RoundingSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Pass-through gradient
        return grad_output
    

class MEC(pl.LightningModule):
    def __init__(self,modalities,models,args=None,preporcess="zscore",data=None,test_set=None,exp = "discrete",stat=None,intervals=[],types= [], alphabet =[],gt = None):
        super(MEC, self).__init__()    
        self.args = args
        self.save_hyperparameters(ignore=["models","modalities","data","gt","stat","test_set"])
        if gt:
            self.gt_ = {k:str(v) for k,v in gt.items()}
            self.hparams.gt_ = self.gt_

        self.modalities= modalities
        self.mod_names= [m.name for m in modalities]
        self.init_stats()
        self.data = data
        self.test_set = test_set
        self.intervals = intervals
        self.sde = DDPM(num_train_timesteps=self.args.num_timesteps)
        self.exp =exp
        self.stat = stat
        self.alphabet = alphabet
     
        self.model_1= models[0].train()
        self.model_2= models[1].train()
        
   
        
        self.gt = gt
        self.preporcess_ = preporcess
        self.intervals_ = self.intervals
        
        self.model_ema_1 = EMA(self.model_1, decay=0.999)  if self.args.ema==1 else None
        self.model_ema_2 = EMA(self.model_2, decay=0.999) if self.args.ema==1 else None

        self.automatic_optimization = False
 
    

    def set_up__priors_diffusion(self):
        if self.args.ema:
            self.frozen_model_1 = deepcopy(self.model_1).requires_grad_(False).eval()
            self.frozen_model_2 = deepcopy(self.model_2).requires_grad_(False).eval()
        else:
            self.frozen_model_1 = deepcopy(self.model_1).requires_grad_(False).eval()
            self.frozen_model_2 = deepcopy(self.model_2).requires_grad_(False).eval()

    def init_stats(self):
        self.gen_step = 0
        self.steps_track={k : { m.name:0 for m in self.modalities} for k in ["d","g","v"]}
        self.rewards_mean = { m.name:0 for m in self.modalities} 
        
    def denoiser_1(self,x=None,t=None,cond=None, old = False, ema =False , drop=0.0 ,cfg=False):
    
        if old ==False:
            if ema:
                return self.model_ema_1.module(x=x,condition=cond,timestep=t,cfg=cfg)
            else:
                return self.model_1(x=x,condition=cond,timestep=t,drop=drop,cfg=cfg)
        else:
            return self.frozen_model_1(x=x,condition=cond,timestep=t,cfg=cfg)
    

    def denoiser_2(self,x=None,t=None,cond=None, old = False,ema= False , drop=0.0,cfg=False ):  

        if old ==False:
            if ema:
                return  self.model_ema_2.module(x=x,condition=cond,timestep=t,cfg=cfg)
            else:
                return self.model_2(x=x,condition=cond,timestep=t,drop=drop,cfg=cfg)
        else:
            return self.frozen_model_2(x=x,condition=cond,timestep=t,cfg=cfg)
        
  
    def update_ema(self,mode) :
        if self.args.ema:
            if mode ==self.mod_names[0]:
                self.model_ema_1.update(self.model_1)
            else:
                self.model_ema_2.update(self.model_2)

    def reward_compute(self,x,x_cond, denoiser,denoiser_2):
        self.model_1.eval()
        self.model_2.eval()
        
        info_e = {"e_x":0,"e_xy":0,"e_y":0,"e_x_new":0 ,"minde":0}
        
        M = 40
       
        t_1 = self.sde.sample_uniform(shape=(M,x.shape[0]) ,eps=self.args.eps[0],max_eps=self.args.eps[1])

        for i in range(M):   
        
            t_x = t_1[i]
            t_y = t_x

            x_t, z, mean, std = self.sde.q_sample(x,t_x)
            y_t,z_y,mean_y,std_y = self.sde.q_sample(x_cond,t_y)

            f,g = self.sde.sde(t_x,x)
            f_y,g_y = self.sde.sde(t_y,x_cond)

            with torch.no_grad():
                eps_cond , eps_uncond =  denoiser(x_t,cond=x_cond,t=t_x, ema=False),denoiser(x_t,cond=None,t=t_x, ema=False)
                eps_cond , eps_uncond = eps_cond.detach() ,eps_uncond.detach()
                eps_uncond_old = denoiser(x_t,cond=None,old=True, t=t_x,ema=False).detach()
                eps_uncond_y = denoiser_2(y_t,cond=None,t=t_y,old=True,ema=False).detach()
            w = 0.5 * self.args.num_timesteps * (g/std)**2
            info_e["e_xy"] +=  w * torch.square(eps_cond - z).sum(dim=1).mean() /M
            info_e["e_y"] += w * torch.square(eps_uncond_y - z_y).sum(dim=1).mean() /M
            info_e["e_x"] +=  w * torch.square(eps_uncond - z).sum(dim=1).mean() /M
            info_e["e_x_new"] +=  w * torch.square(eps_uncond_old - z).sum(dim=1).mean() /M
            info_e["minde"] += w * torch.square(eps_uncond - eps_cond).sum(dim=1).mean() /M

        mi_ = { "mi_theo":info_e["e_x"]- info_e["e_xy"], "mi_theo_new":info_e["e_x_new"]- info_e["e_xy"],}
        return mi_, info_e 
