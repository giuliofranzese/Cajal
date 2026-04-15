import torch
import pytorch_lightning as pl
from src.diffusion.sde import *
from src.diffusion.utils import EMA
from copy import deepcopy

#from src.diffusion.ddpm_n import DDPM 

from src.diffusion.ddpm import DDPM
from diffusers import DDIMScheduler
from src.discrete.utils import *

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import SCOT.src.evals as evals
import pandas as pd
import scanpy as sc
from scTopoGAN.Qualitative_Metrics import evaluate_neighbors
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
        self.sde = DDPM(T=self.args.num_timesteps,beta_max=self.args.betas[1],beta_min=self.args.betas[0])
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
        if self.args.reply_buffer:
            self.reply_buffer = {m.name:{"x":torch.Tensor(),"cond":torch.Tensor()} for m in self.modalities}
    

    def set_up__priors_diffusion(self):
        if self.args.ema:
            self.model_1.load_state_dict(self.model_ema_1.module.state_dict())
            self.model_2.load_state_dict(self.model_ema_2.module.state_dict())
            # self.model_ema_1.module.train().requires_grad_(False)
            # self.model_ema_2.module.train().requires_grad_(False)
            self.frozen_model_1 = deepcopy(self.model_1).requires_grad_(False).eval()
            self.frozen_model_2 = deepcopy(self.model_2).requires_grad_(False).eval()
        else:
            self.frozen_model_1 = deepcopy(self.model_1).requires_grad_(False).eval()
            self.frozen_model_2 = deepcopy(self.model_2).requires_grad_(False).eval()
        # self.model_1.add_control_net()
        # self.model_2.add_control_net()

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
       
        # #t_1 = self.sde.sample_uniform(shape=(M,x.shape[0]) ,eps=self.args.eps[0],max_eps=self.args.eps[1])
        # sch = DDIMScheduler(num_train_timesteps=1000, steps_offset=1)
        # sch.set_timesteps(num_inference_steps=M)
        # t = sch.timesteps
   
        #t_1 = torch.tensor(t,device=x.device)
        t_1 , weights = self.sde.select_timesteps(n=M)
        for i in range(M):   
        
           # t_x = t_1[i]
            t_x = t_1[i].repeat(x.shape[0])
            w = weights[i].repeat(x.shape[0])
            t_y = t_x

            x_t, z, mean, std = self.sde.q_sample(x,t_x)
            y_t,z_y,mean_y,std_y = self.sde.q_sample(x_cond,t_y)

            # f,g = self.sde.sde(t_x,x)
            # f_y,g_y = self.sde.sde(t_y,x_cond)
            #w = self.sde.snr_t(t_x)

            with torch.no_grad():
                eps_cond , eps_uncond =  denoiser(x_t,cond=x_cond,t=t_x, ema=False),denoiser(x_t,cond=None,t=t_x, ema=False)
                eps_cond , eps_uncond = eps_cond.detach() ,eps_uncond.detach()
                eps_uncond_old = denoiser(x_t,cond=None,old=True, t=t_x,ema=False).detach()
                eps_uncond_y = denoiser_2(y_t,cond=None,t=t_y,old=True,ema=False).detach()
           # w = 0.5 * self.args.num_timesteps * (g/std)**2
            info_e["e_xy"] +=  w * torch.square(eps_cond - z).sum(dim=1).mean() /M
            info_e["e_y"] += w * torch.square(eps_uncond_y - z_y).sum(dim=1).mean() /M
            info_e["e_x"] +=  w * torch.square(eps_uncond - z).sum(dim=1).mean() /M
            info_e["e_x_new"] +=  w * torch.square(eps_uncond_old - z).sum(dim=1).mean() /M
            info_e["minde"] += w * torch.square(eps_uncond - eps_cond).sum(dim=1).mean() /M

        mi_ = { "mi_theo":info_e["e_x"]- info_e["e_xy"], "mi_theo_new":info_e["e_x_new"]- info_e["e_xy"],}
        return mi_, info_e 



    def train_denoiser(self,x,x_cond, denoiser,mod,opt ):
        i =0 if mod == self.modalities[0].name else 1

        self.model_1.train()
        self.model_2.train()

        opt.zero_grad()
        if  self.gen_step <self.args.warmup  :
            uncond_p = self.args.uncond_p[0]
            grad_scale = self.args.grad_scale[0]
        else :
            uncond_p = self.args.uncond_p[1]
            grad_scale = self.args.grad_scale[1]

        t = self.sde.sample_uniform(shape=(x.size(0),) )
        #print("t",t)
   
        x_t, z, mean, std = self.sde.q_sample(x,t) 

        e_c = denoiser(x_t,cond=x_cond,t=t ,drop= uncond_p )#  )
        loss =   torch.square(z - e_c).mean()
        (grad_scale * loss).backward()
        n1 =torch.nn.utils.clip_grad_norm_(self.model_1.parameters(), self.args.clip_norm[0] )
        n2 = torch.nn.utils.clip_grad_norm_(self.model_2.parameters(), self.args.clip_norm[0]  )
        opt.step()
        opt.zero_grad()
        self.update_ema(mod)
        self.steps_track["d"][mod] +=1    
        self.log_scalars(label="Reward_update_{}".format(mod),log={"norm":{"n1":n1,"n2":n2},
                                                                   "losses":loss.mean()},step =  self.steps_track["d"][mod])
        return loss 



    def gen_trajectories(self,x,y_cond,denoiser_gen,denoiser_reward,mod_gen):
        
        self.model_1.eval()
        self.model_2.eval()
        i = 0 if  self.modalities[0].name == mod_gen else 1
        with torch.no_grad():
            x_traj , log_probs, timesteps ,timesteps_prev  = self.sde.sample_from_model_with_logprob(score_function=denoiser_gen,
                                              x0=torch.randn_like(x),
                                               c=y_cond,
                                              guidance=self.args.guidance,
                                              num_steps=self.args.num_steps_train,
                                              clip_gen= self.args.clip_gen if self.intervals_[0] !=None else False ,
                                              intervals= self.intervals_[i],
                                              eta=self.args.eta,
                                              ema=False) 
  
            x_gen =  x_traj[-1]

            x_gen = self.postporcess(x_gen,i,proj=self.args.neighbor)
            x_gen = self.preprocess(x_gen,i)
         
            if self.args.reward =="vlb":
                t  = self.sde.sample_uniform(shape=(self.args.mc_steps,)  ,eps=self.args.eps[0],max_eps=self.args.eps[1])
                #t = torch.arange(1, self.args.num_timesteps-1, device=self.device)
                w = torch.ones_like(t)
                if self.trainer.world_size > 1:
                    t,w = self.all_gather(t)[0],self.all_gather(w)[0]
            # else:
            #     snr,w  = logistic_integrate(self.args.mc_steps,loc=1.0,scale=2.0,clip=3.0, device=self.device)
            #     t = self.sde.logsnr2t(snr)
            #     w = torch.ones_like(t)
            #     if self.trainer.world_size > 1:
            #         t,w = self.all_gather(t)[0],self.all_gather(w)[0]
                 
            rewards = - self.sde.nll(y_cond,cond = x_gen,model=denoiser_reward,t= t,w =w,reduce = self.args.reduce,weighted = self.args.weighted,loss_type=self.args.loss_type,sampling=self.args.sampling).detach()
    
   
        x_traj = torch.stack(x_traj, dim=1).detach() 
        
        log_probs = torch.stack(log_probs, dim=1).detach()   # (batch_size, num_steps, 1)
  
        timesteps = torch.stack(timesteps, dim=1).detach()  

        timesteps_prev = torch.stack(timesteps_prev, dim=1).detach()   

        sample= {       "cond": y_cond,
                        "timesteps": timesteps,
                        "timesteps_prev":timesteps_prev,
                        "latents": x_traj[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": x_traj[:, 1:],  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "rewards": rewards.view(-1,1)}

        if self.gen_step %2500==0:
                mi,e  = self.reward_compute(x=y_cond,x_cond=x_gen,denoiser=denoiser_reward, denoiser_2=denoiser_gen)
                mi = {k: mi[k].mean()  for k in mi.keys()}
                e = {k: e[k].mean()  for k in e.keys()}
                self.log_scalars(label="info_measures{}".format(mod_gen),log={"mi":mi,"e":e},step =  self.steps_track["g"][mod_gen])
        return sample, x_gen
    


    def log_scalars(self,log,label,step):
        for k in log.keys():
            if isinstance( log[k],dict ):
                self.logger.experiment.add_scalars(label +"/"+ k, log[k] , global_step=step)
            else:
                self.logger.experiment.add_scalar(label +"/"+ k, log[k] , global_step=step)


    def policy_gradient_update_dpok(self,samples,denoiser_gen,opt_gen,mod_gen):
        self.model_1.train()
        self.model_2.train()
        
        mod = 0 if mod_gen == self.mod_names[0] else 1
        batch_size = self.args.batch_size_train
        num_steps = self.args.num_steps_train
        total_batch_size = samples["cond"].shape[0]
   #     print(f"Rank: {self.global_rank}, World Size: {self.trainer.world_size}")

        if self.args.advantage:
            rewards = samples["rewards"]
            if self.trainer.world_size>1:
                rewards_global = self.all_gather(rewards).view(-1,rewards.shape[1])
            else:
                rewards_global =rewards
            samples["advantages"]  = (samples["rewards"]- rewards_global.mean() )  /( rewards_global.std() +1e-8)
            self.log_scalars(label="PG_update{}".format(mod_gen),log={"advantage_glob": ( (rewards_global- rewards_global.mean() )  /( rewards_global.std() +1e-8) ).mean(),
                                                    "reward_mean":rewards_global.mean(),
                                                    "reward_std":rewards_global.std(),
                                                   },step =  self.gen_step)
        else:
            samples["advantages"] = samples["rewards"]

            self.log_scalars(label="PG_update{}".format(mod_gen),log={"reward_mean":samples["rewards"].mean(),
                                                    "reward_std":samples["rewards"].std(),
                                                   },step =  self.gen_step)
     
            
        samples["rewards"] = samples["rewards"].unsqueeze(1).repeat(1, self.args.num_steps_train, 1)
        samples["advantages"] = samples["advantages"].unsqueeze(1).repeat(1, self.args.num_steps_train, 1)

        samples["cond"] = samples["cond"].unsqueeze(1).repeat(1, self.args.num_steps_train, 1)


        #conds = samples["cond"]
        #samples["cond"] = torch.arange(samples["cond"].shape[0] *self.args.num_steps_train, device=self.device).reshape(-1, 1)

        samples_timestep = {
            k : samples[k].view(total_batch_size*num_steps,*samples[k].shape[2:]) for k in samples.keys()
        }
        m=0
        for _ in   range(self.args.nb_update_pg) :
            losses = 0
            perm = torch.randperm(total_batch_size*self.args.num_steps_train, device=self.device)#[:batch_size*self.args.num_grad_accummulate]
            
            samples_batched ={
                k : samples_timestep[k][perm].reshape(-1, batch_size, *samples_timestep[k].shape[1:])  for k in samples_timestep.keys()
            }

       
       
            ratios = log_probs_olds =log_probs =losses = advantages_l  = rewards_l =kl_reg_l= 0

            nb_batches = samples_batched["latents"].shape[0]
     
            for j in range(nb_batches) :
                x_t = samples_batched["latents"][j]
                next_x_t = samples_batched["next_latents"][j]
                # cond_idx = samples_batched["cond"][j]//self.args.num_steps_train
                # cond = conds[cond_idx.long()]
                cond = samples_batched["cond"][j]
                t = samples_batched["timesteps"][j]
                t_prev = samples_batched["timesteps_prev"][j]
                log_prob_old = samples_batched["log_probs"][j]
                advantages = samples_batched["advantages"][j]
                rewards = samples_batched["rewards"][j]
           

                with torch.enable_grad():

                        if self.args.guidance!=0 and self.args.guidance!=1.0:
                            noise_pred,uncond = denoiser_gen(x_t,cond = cond,t=t),denoiser_gen(x_t,cond = None,t=t)
                            if self.args.train_uncond==0:
                                uncond = uncond.detach()
                            noise_pred_step = uncond + self.args.guidance * (noise_pred-uncond)
                        elif self.args.guidance==1.0:
                            noise_pred = denoiser_gen(x_t,cond=cond,t=t)
                            noise_pred_step = noise_pred
                        else:
                            noise_pred = denoiser_gen(x_t,cond=None,t=t)
                            noise_pred_step = noise_pred
          
                # _,log_prob = self.sde.step_forward_logprob(model_output=noise_pred_step,timestep=t,sample=x_t,next_sample=next_x_t,eta=1.0)
                

                _,log_prob = self.sde.ddim_step_with_logprob_forward(noise_pred_step,sample=x_t,sample_next=next_x_t,timestep_prev=t_prev,
                timestep=t,clip_gen=False ,eta=self.args.eta , intervals= self.intervals_[mod])

                log_prob = log_prob.view(log_prob.shape[0],1)
             
                if self.args.kl_weight[mod] > 0:  
                    
                    if self.args.train_uncond==1 and self.args.guidance!=0:
                        noise_pred_old = denoiser_gen(x_t,cond =None,t=t,old=True)
                        kl_reg = ((noise_pred_step - noise_pred_old)**2  ).mean()
                    else:
                        noise_pred_old = denoiser_gen(x_t,cond = None,t= t,old=True) ## Unconditional model and new conditional model
                        kl_reg = ((noise_pred - noise_pred_old)**2  ).mean()
                else:
                    kl_reg = 0

                if self.args.pg_importance_sampling==1:
                        ratio = torch.exp(log_prob -log_prob_old )
                        clipped_loss = - advantages * torch.clamp(
                                ratio,
                                1.0 - self.args.clip_range_pg,
                                1.0 + self.args.clip_range_pg,
                            ).float().reshape([self.args.batch_size_train, 1])
                        loss = clipped_loss
                else:                        
                        loss = - advantages * log_prob 

                loss = self.args.reward_weight[mod] * loss.mean() +  self.args.kl_weight[mod] * kl_reg
                ( loss/ self.args.num_grad_accummulate).backward()  
                ratios += ratio.mean()/self.args.num_grad_accummulate
                losses += loss.mean()/self.args.num_grad_accummulate
                log_probs += log_prob.mean()/self.args.num_grad_accummulate
                log_probs_olds += log_prob_old.mean()/self.args.num_grad_accummulate
                advantages_l += advantages.mean()/self.args.num_grad_accummulate
                rewards_l += rewards.mean()/self.args.num_grad_accummulate
                kl_reg_l+= kl_reg/self.args.num_grad_accummulate

                if (j+1)%self.args.num_grad_accummulate==0:
                
                    self.steps_track["g"][mod_gen] +=1
                    n1 =torch.nn.utils.clip_grad_norm_(self.model_1.parameters(), self.args.clip_norm[0] )
                    n2 = torch.nn.utils.clip_grad_norm_(self.model_2.parameters(), self.args.clip_norm[0]  )

                    self.log_scalars(label="PG_update{}".format(mod_gen),log={"loss":losses,"kl_reg_l":kl_reg_l,
                                                            "norm":{"norm1":n1,"norm2":n2 },
                                                            "advantage":advantages_l,
                                                            "ratio":ratios,
                                                            "log_prob":{"log_prob_new":log_probs,"log_probs_olds":log_probs_olds}
                                                            },step =  self.steps_track["g"][mod_gen])
            
                    opt_gen.step()
                    opt_gen.zero_grad()
                    self.update_ema(mod_gen)
                    ratios =log_probs_olds =log_probs =losses = advantages_l  = rewards_l = kl_reg_l = 0
                    m=m+1

    





    def gen_dpok(self,x,y,denoiser_gen, denoiser_reward,opt, mod_gen):
 
        samples = []
        gen = []
 
        for y_cond in y.view(-1, self.args.batch_size_sample, *y.shape[1:]) :
                traj, gen_hat = self.gen_trajectories(x[:self.args.batch_size_sample],y_cond,denoiser_gen,denoiser_reward,mod_gen )
                gen.append(gen_hat)
                samples.append(traj)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        if self.args.method =="ddpo":
            self.policy_gradient_update_ddpo(samples,denoiser_gen,opt,mod_gen)
        else:
            self.policy_gradient_update_dpok(samples,denoiser_gen,opt,mod_gen)
            
        del samples
        gen  = torch.cat(gen,dim=0)
        return { "cond": y,
                "out":gen}
    








    # def update_models(self, x, x_cond, denoiser, mod, opt, nb, step=0):
    #     if self.args.reply_buffer:
    #         mod_buffer = self.reply_buffer[mod]

    #         # Detach inputs for buffer usage
    #         x_new = x.detach()
    #         x_cond_new = x_cond.detach()

    #         # Initialize buffer if empty
    #         if "x" not in mod_buffer or len(mod_buffer["x"]) == 0:
    #             mod_buffer["x"] = x_new
    #             mod_buffer["cond"] = x_cond_new
    #             x_total = x_new
    #             x_cond_total = x_cond_new
    #         else:
    #             buffer_size = mod_buffer["x"].size(0)
    #             num_replay = min(self.args.size_buffer, buffer_size)

    #             # Inverse annealing: start mostly random, shift to more recent
    #             # anneal_steps = 2000
    #             # start_recent_ratio = 0.5
    #             # max_recent_ratio = 0.9
    #            # step = max(0, self.gen_step - self.args.warmup)
    #             recent_ratio = 1- self.args.buffer_old
    #             # recent_ratio = min(
    #             #     start_recent_ratio + (step / anneal_steps) * (max_recent_ratio - start_recent_ratio),
    #             #     max_recent_ratio,
    #             # )

    #             # Adjust for the fact that x_new is already recent
    #             total_batch_size = num_replay + x_new.size(0)
    #             effective_recent_ratio = max(0.0, recent_ratio - (x_new.size(0) / total_batch_size))
    #             num_recent = int(effective_recent_ratio * num_replay)
    #             num_random = num_replay - num_recent

    #             # Get recent samples from end of buffer
    #             if num_recent > 0:
    #                 recent_indices = torch.arange(buffer_size - num_recent, buffer_size, device=self.device)
    #             else:
    #                 recent_indices = torch.tensor([], dtype=torch.long, device=self.device)

    #             # Random samples from older buffer (exclude recent)
    #             if buffer_size > num_recent and num_random > 0:
    #                 rand_pool = torch.arange(0, buffer_size - num_recent, device=self.device)
    #                 rand_indices = rand_pool[torch.randperm(len(rand_pool))[:num_random]]
    #             else:
    #                 rand_indices = torch.tensor([], dtype=torch.long, device=self.device)

    #             selected_indices = torch.cat([recent_indices, rand_indices])
    #             x_replay = mod_buffer["x"][selected_indices]
    #             cond_replay = mod_buffer["cond"][selected_indices]

    #             # Combine for training
    #             x_total = torch.cat([x_replay, x_new], dim=0)
    #             x_cond_total = torch.cat([cond_replay, x_cond_new], dim=0)

    #             # Update buffer
    #             mod_buffer["x"] = torch.cat([mod_buffer["x"], x_new], dim=0)
    #             mod_buffer["cond"] = torch.cat([mod_buffer["cond"], x_cond_new], dim=0)

    #             # Truncate buffer if needed
    #             max_size = self.args.size_buffer * 10
    #             if mod_buffer["x"].size(0) > max_size:
    #                 perm = torch.randperm(mod_buffer["x"].size(0), device=self.device)
    #                 mod_buffer["x"] = mod_buffer["x"][perm[:max_size]]
    #                 mod_buffer["cond"] = mod_buffer["cond"][perm[:max_size]]
    #                 # mod_buffer["x"] = mod_buffer["x"][-max_size:]
    #                 # mod_buffer["cond"] = mod_buffer["cond"][-max_size:]
    #     else:
    #         # No replay buffer
    #         x_total = x
    #         x_cond_total = x_cond

    #     # Shuffle and train
    #     for _ in range(nb):
    #         rand_idx = torch.randperm(x_total.size(0), device=self.device)
    #         x_shuf = x_total[rand_idx]
    #         x_cond_shuf = x_cond_total[rand_idx]

    #         for x_batch, cond_batch in zip(
    #             x_shuf.split(self.args.batch_size_update),
    #             x_cond_shuf.split(self.args.batch_size_update)
    #         ):
    #             if x_batch.size(0) < self.args.batch_size_update:
    #                 continue
    #             self.train_denoiser(
    #                 x=x_batch.detach(), x_cond=cond_batch.detach(),
    #                 denoiser=denoiser, mod=mod, opt=opt
    #             )
    # def update_models(self, x, x_cond, denoiser, mod, opt, nb, step=0):
    #     if self.args.reply_buffer:
    #         mod_buffer = self.reply_buffer[mod]

    #         # Detach inputs for buffer usage
    #         x_new = x.detach()
    #         x_cond_new = x_cond.detach()

    #         # Handle empty buffer
    #         if "x" not in mod_buffer or len(mod_buffer["x"]) == 0:
    #             mod_buffer["x"] = x_new
    #             mod_buffer["cond"] = x_cond_new
    #             x_total = x_new
    #             x_cond_total = x_cond_new
    #         else:
    #             # Step 1: Sample from buffer
    #             num_replay = min(self.args.size_buffer, mod_buffer["x"].size(0))

    #             rand_indices = torch.randperm(mod_buffer["x"].size(0), device=self.device)[:num_replay]

    #             x_replay = mod_buffer["x"][rand_indices]
    #             cond_replay = mod_buffer["cond"][rand_indices]

    #             # Step 2: Combine old + new for training
    #             x_total = torch.cat([x_replay, x_new], dim=0)
    #             x_cond_total = torch.cat([cond_replay, x_cond_new], dim=0)

    #             # Step 3: Add new data to buffer
    #             mod_buffer["x"] = torch.cat([mod_buffer["x"], x_new], dim=0)
    #             mod_buffer["cond"] = torch.cat([mod_buffer["cond"], x_cond_new], dim=0)

    #             # Step 4: Truncate buffer if needed
    #             max_size = self.args.size_buffer * 10
    #             if mod_buffer["x"].size(0) > max_size:
    #                 perm = torch.randperm(mod_buffer["x"].size(0), device=self.device)
    #                 mod_buffer["x"] = mod_buffer["x"][perm[:max_size]]
    #                 mod_buffer["cond"] = mod_buffer["cond"][perm[:max_size]]
    #                 # mod_buffer["x"] = mod_buffer["x"][-max_size:]
    #                 # mod_buffer["cond"] = mod_buffer["cond"][-max_size:]
    #     else:
    #         # If no replay buffer is used
    #         x_total = x
    #         x_cond_total = x_cond

    #     # Step 5: Shuffle and train
    #     for _ in range(nb):
    #         rand_idx = torch.randperm(x_total.size(0), device=self.device)
    #         x_shuf = x_total[rand_idx]
    #         x_cond_shuf = x_cond_total[rand_idx]

    #         for x_batch, cond_batch in zip(
    #             x_shuf.split(self.args.batch_size_update),
    #             x_cond_shuf.split(self.args.batch_size_update)
    #         ):
    #             if x_batch.size(0) < self.args.batch_size_update:
    #                 continue
    #             self.train_denoiser(
    #                 x=x_batch.detach(), x_cond=cond_batch.detach(),
    #                 denoiser=denoiser, mod=mod, opt=opt
    #             )   
    


    def update_models(self, x, x_cond, denoiser, mod, opt, nb, step=0):
        if self.args.reply_buffer:
            mod_buffer = self.reply_buffer[mod]

            # Detach inputs for buffer usage
            x_new = x.detach()
            x_cond_new = x_cond.detach()

            # Handle empty buffer
            if "x" not in mod_buffer or len(mod_buffer["x"]) == 0:
                mod_buffer["x"] = x_new
                mod_buffer["cond"] = x_cond_new
                x_total = x_new
                x_cond_total = x_cond_new
            else:
                # Step 1: Sample from buffer
                num_replay = min(self.args.size_buffer, mod_buffer["x"].size(0))
                rand_indices = torch.randperm(mod_buffer["x"].size(0), device=self.device)[:num_replay]
                x_replay = mod_buffer["x"][rand_indices]
                cond_replay = mod_buffer["cond"][rand_indices]

                # Step 2: Combine old + new for training
                x_total = torch.cat([x_replay, x_new], dim=0)
                x_cond_total = torch.cat([cond_replay, x_cond_new], dim=0)

                # Step 3: Add new data to buffer
                mod_buffer["x"] = torch.cat([mod_buffer["x"], x_new], dim=0)
                mod_buffer["cond"] = torch.cat([mod_buffer["cond"], x_cond_new], dim=0)

                # Step 4: Truncate buffer if needed
                max_size = self.args.size_buffer * 10
                if mod_buffer["x"].size(0) > max_size:
                    perm = torch.randperm(mod_buffer["x"].size(0), device=self.device)
                    mod_buffer["x"] = mod_buffer["x"][perm[:max_size]]
                    mod_buffer["cond"] = mod_buffer["cond"][perm[:max_size]]
        else:
            # If no replay buffer is used
            x_total = x
            x_cond_total = x_cond

        # Step 5: Shuffle and train
        for _ in range(nb):
            rand_idx = torch.randperm(x_total.size(0), device=self.device)
            x_shuf = x_total[rand_idx]
            x_cond_shuf = x_cond_total[rand_idx]

            for x_batch, cond_batch in zip(
                x_shuf.split(self.args.batch_size_update),
                x_cond_shuf.split(self.args.batch_size_update)
            ):
                if x_batch.size(0) < self.args.batch_size_update:
                    continue
                self.train_denoiser(
                    x=x_batch.detach(), x_cond=cond_batch.detach(),
                    denoiser=denoiser, mod=mod, opt=opt
                )


           
    # def update_models(self,x,x_cond,denoiser,mod,opt ,nb):
    #     total_buffer_size = self.args.buffer_recent + self.args.size_buffer
    #     if self.args.reply_buffer:
    #         x_new = x
    #         x_cond_new = x_cond
    #         mod_buffer = self.reply_buffer[mod]
    #         if len(mod_buffer["x"]) > total_buffer_size:

    #             # Step 1: Sample old data
    #             if self.args.buffer_recent>0:
    #                 x_replay = mod_buffer["x"][-self.args.buffer_recent:]
    #                 cond_replay = mod_buffer["cond"][-self.args.buffer_recent:]
    #                 x_new = torch.cat([x_replay, x_new], dim=0)
    #                 x_cond_new = torch.cat([cond_replay, x_cond_new], dim=0)    

    #             rand_indices = torch.randperm(mod_buffer["x"].size(0)-self.args.buffer_recent, device=self.device)[:self.args.size_buffer]  
    #             x_replay = mod_buffer["x"][rand_indices]
    #             cond_replay = mod_buffer["cond"][rand_indices]


    #             mod_buffer["x"] = torch.cat([mod_buffer["x"].to(self.device), x], dim=0)
    #             mod_buffer["cond"] = torch.cat([mod_buffer["cond"].to(self.device), x_cond], dim=0)
                
    #             # Step 2: Combine with current batch
    #             x_new = torch.cat([x_replay, x_new], dim=0)
    #             x_cond_new = torch.cat([cond_replay, x_cond_new], dim=0)

    #             # Step 3: Update buffer with new samples
                

    #             # Step 4: Truncate with shared permutation to maintain alignment
    #             max_size = self.args.size_buffer * 10
    #             if mod_buffer["x"].size(0) > max_size:
    #                 perm = torch.randperm(mod_buffer["x"].size(0), device=self.device)[:max_size]
    #                 keep_indices, _ = torch.sort(perm)
    #                 mod_buffer["x"] = mod_buffer["x"][keep_indices]
    #                 mod_buffer["cond"] = mod_buffer["cond"][keep_indices]
                
    #         else:
    #             mod_buffer["x"] = torch.cat([mod_buffer["x"].to(self.device),x.detach()])
    #             mod_buffer["cond"] = torch.cat([mod_buffer["cond"].to(self.device),x_cond])
    #         x = x_new
    #         x_cond = x_cond_new
    #     # print(x.shape)
    #     # print(x_cond_new.shape)
    #     for _ in range(nb):
    #         rand_idx = torch.randperm(x_cond.size(0), device=self.device)
    #         x_rand, x_cond_rand = x[rand_idx], x_cond[rand_idx]
    #         batched_x = x_rand.view(-1, self.args.batch_size_update, *x.shape[1:])
    #         batched_cond = x_cond_rand.view(-1, self.args.batch_size_update, *x_cond.shape[1:]) 
    #         k = 0
    #         for x_batch ,cond_batch  in zip(batched_x ,  batched_cond):
    #             self.train_denoiser(x=x_batch.detach(),x_cond=cond_batch.detach(),denoiser=denoiser,mod=mod,opt=opt )
    #             k =k+1
    #  #   print("did: "+str(k))
 
    def run_one_way(self,training_method,gen,cond,mod_reward,mod_gen, denoiser_gen,denoiser_reward,opt_gen,opt_reward):
            
                out = training_method( denoiser_gen=denoiser_gen,
                                                        x=gen,
                                                        y=cond,
                                                        denoiser_reward= denoiser_reward,
                                                        opt= opt_gen , 
                                                        mod_gen = mod_gen)                    
                self.update_models(x= out["cond"],x_cond=out["out"],
                                   denoiser = denoiser_reward,
                                   opt=opt_reward,
                                   mod = mod_reward,
                                   nb = self.args.nb_update_reward )       

    def save_custom_checkpoint(self):
        """Saves a checkpoint using the default Trainer checkpoint directory."""
        if self.trainer:
                # Generate the path based on global step
                versioned_dir = self.trainer.logger.log_dir  # e.g., checkpoints/version_0
                checkpoint_dir = f"{versioned_dir}/checkpoints"  # Ensure we store in the checkpoints folder

                checkpoint_path = f"{checkpoint_dir}/checkpoint-step-{self.gen_step:06d}.ckpt"
                # Save the checkpoint
                self.trainer.save_checkpoint(checkpoint_path)
              #  print(f"Custom checkpoint saved at: {checkpoint_path}")


    def training_step(self, batch, batch_idx):
        self.sde.device = self.device
        self.gen_step+=1
        opt1, opt2 = self.optimizers()
        if self.gen_step ==self.args.warmup:
            self.set_up__priors_diffusion()
            self.save_custom_checkpoint()
            for param_group in opt1.param_groups:
                param_group['lr'] = self.args.lr[1]
            for param_group in opt2.param_groups:
                param_group['lr'] = self.args.lr[1]
        elif self.gen_step ==self.args.max_step:
            self.trainer.should_stop = True    
        self.model_1.eval()
        self.model_2.eval()
        

        x1, x2 = self.preprocess( batch[0] , 0)  ,self.preprocess(batch[1],1)

        training_method = self.gen_direct if self.args.method =="direct" else self.gen_dpok

        if self.gen_step >= self.args.warmup: 
                
                
                self.run_one_way(training_method=training_method,gen=x2,cond=x1,  mod_gen=self.mod_names[1],denoiser_gen=self.denoiser_2,
                                mod_reward=self.mod_names[0],denoiser_reward=self.denoiser_1,
                                opt_gen=opt2,opt_reward=opt1) 
                
                self.run_one_way(training_method=training_method,gen=x1,cond=x2,  mod_gen=self.mod_names[0],denoiser_gen=self.denoiser_1,
                                mod_reward=self.mod_names[1],denoiser_reward=self.denoiser_2,
                                opt_gen=opt1,opt_reward=opt2)
        else:
        
            self.train_denoiser(x=x1[:self.args.batch_size_update],x_cond=x2[:self.args.batch_size_update], denoiser=self.denoiser_1,mod=self.mod_names[0],opt=opt1 )
            self.train_denoiser(x=x2[:self.args.batch_size_update],x_cond=x1[:self.args.batch_size_update], denoiser=self.denoiser_2,mod=self.mod_names[1],opt=opt2 )
        
        
        # if self.gen_step >= 20000:
        #     if self.gen_step %20000==0:
        #         self.save_custom_checkpoint()
        #        # self.log_step_2()
        
        if (self.gen_step % self.args.log_step ==1  and self.gen_step>self.args.warmup) :#or self.gen_step %1000 ==0:
                if self.exp=="snare":
                    self.log_step_1()
                else:
                    self.log_step_2()

        elif self.gen_step < self.args.warmup:
                self.model_1.eval()
                self.model_2.eval()
                if self.exp=="snare" and self.gen_step % 1000 ==0:
                    self.log_step_warmup_1()
                elif self.gen_step % 10000 ==0:
                    self.log_step_warmup_2()
          

        

    import torch

    def find_nearest_neighbors(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        # Compute the squared Euclidean distance between each vector in set1 and each vector in set2
        set1 = set1.to(torch.float32)
        set2 = set2.to(torch.float32).to(set1.device)
        distances = torch.cdist(set1, set2, p=2)
        # Find the indices of the nearest neighbors in set2 for each vector in set1
        nearest_indices = torch.argmin(distances, dim=1)
        # Retrieve the nearest neighbor vectors from set2
        nearest_neighbors = set2[nearest_indices]
        return nearest_neighbors


    def eval_coupling_2(self,guidance,nb_step,eta=1.0,clip_gen=True):
        self.model_1.eval()
        self.model_2.eval()
        x1,x2 =self.data
        x1 = self.preprocess(x1,0).to(self.device)
        x2 = self.preprocess(x2,1).to(self.device)
            # x1 = self.preprocess(self.test_set["x1"]["data"]  ,0).to(self.device)
            # x2 = self.preprocess(self.test_set["x2"]["data"]  ,1).to(self.device)
        x1_Celltype = self.test_set["x1"]["Celltype"]
        x1_Subcelltype = self.test_set["x1"]["Subcelltype"]
        x2_Celltype = self.test_set["x2"]["Celltype"]
        x2_Subcelltype = self.test_set["x2"]["Subcelltype"]
  
        x_T_1 = torch.randn(*x1.shape,device = self.device)
        x_T_2 = torch.randn(*x2.shape,device = self.device)
        ema= False
        with torch.no_grad(): 
            x_gen_hat_1 = self.sde.sample_from_model(score_function=self.denoiser_1,
                                                                x0=x_T_1, 
                                                                 c=x2,guidance=guidance,eta=eta,
                                                                clip_gen= clip_gen,
                                                            intervals=self.intervals_[0],
                                                                num_steps=nb_step,ema=ema) 
                                    
            x_gen_hat_2 = self.sde.sample_from_model(score_function=self.denoiser_2,
                                                                x0=x_T_2,eta=eta,
                                                                 c=x1,guidance=guidance,     
                                                                clip_gen= clip_gen,
                                                            intervals=self.intervals_[1],
                                                                num_steps=nb_step,ema=ema) 

            x_gen_hat_1 = self.postporcess(x_gen_hat_1,mod =0)
            x_gen_hat_2 = self.postporcess(x_gen_hat_2,mod =1)

            x1_ = self.postporcess(x1,mod =0)
            x2_ = self.postporcess(x2,mod =1)

            x_gen_hat_1_proj = self.find_nearest_neighbors(x_gen_hat_1,x1_)
            x_gen_hat_2_proj = self.find_nearest_neighbors(x_gen_hat_2,x2_)

            x_gen_hat_1 = x_gen_hat_1.cpu().numpy()
            x_gen_hat_2 = x_gen_hat_2.cpu().numpy()
            x1_ = x1_.cpu().numpy()
            x2_ = x2_.cpu().numpy() 
                                    

            x_gen_hat_1_proj = x_gen_hat_1_proj.cpu().numpy()
            x_gen_hat_2_proj = x_gen_hat_2_proj.cpu().numpy()

            CTM_1 = evaluate_neighbors(x_gen_hat_1, x1_, x2_Celltype, x1_Celltype, k=5)
            SCTM_1 = evaluate_neighbors(x_gen_hat_1, x1_, x2_Subcelltype, x1_Subcelltype, k=5)

            CTM_2 = evaluate_neighbors(x_gen_hat_2, x2_, x1_Celltype, x2_Celltype, k=5)
            SCTM_2 = evaluate_neighbors(x_gen_hat_2, x2_, x1_Subcelltype, x2_Subcelltype, k=5)

            CTM_1_proj = evaluate_neighbors(x_gen_hat_1_proj, x1_, x2_Celltype, x1_Celltype, k=5)
            SCTM_1_proj  = evaluate_neighbors(x_gen_hat_1_proj, x1_, x2_Subcelltype, x1_Subcelltype, k=5)

            CTM_2_proj  = evaluate_neighbors(x_gen_hat_2_proj, x2_, x1_Celltype, x2_Celltype, k=5)
            SCTM_2_proj  = evaluate_neighbors(x_gen_hat_2_proj, x2_, x1_Subcelltype, x2_Subcelltype, k=5)

            # print("gen_step",self.gen_step,label_config,guidance,"CTM:", CTM_1, CTM_2)
            # print("gen_step",self.gen_step,label_config,guidance,"SCTM:",SCTM_1, SCTM_2)

            # print("gen_step",self.gen_step,label_config,guidance,"CTM:", CTM_1_proj, CTM_2_proj)
            # print("gen_step",self.gen_step,label_config,guidance,"SCTM:",SCTM_1_proj, SCTM_2_proj)
            fig1 = self.plot_umap_fig(x_gen_hat_2,x2_,self.test_set["x2"]["Celltype"],self.test_set["x2"]["Subcelltype"],N=10000,both=True)

            return { "CTM": [CTM_1, CTM_2],
            "SCTM":[SCTM_1, SCTM_2],
            "CTM_proj":[ CTM_1_proj, CTM_2_proj],
            "SCTM_proj":[SCTM_1_proj, SCTM_2_proj]
            },fig1

            # self.logger.experiment.add_scalars("config_{}/CTM{}/".format(label_config,guidance), 
            #                                                                     {"x1": CTM_1,
            #                                                                     "x2":CTM_2 },self.gen_step )
            # self.logger.experiment.add_scalars("config_{}/SCTM{}/".format(label_config,guidance), 
            #                                                                     {"x1": SCTM_1,
            #                                                                     "x2":SCTM_2 },self.gen_step )

            # self.logger.experiment.add_scalars("config_{}_proj/CTM{}/".format(label_config,guidance), 
            #                                                                     {"x1": CTM_1_proj,
            #                                                                     "x2":CTM_2_proj },self.gen_step )
            # self.logger.experiment.add_scalars("config_{}_proj/SCTM{}/".format(label_config,guidance), 
            #                                                                     {"x1": SCTM_1_proj,
            #                                                                     "x2":SCTM_2_proj },self.gen_step )

            # fig1 = self.plot_umap_fig(x_gen_hat_1,x1_,self.test_set["x1"]["Celltype"],self.test_set["x1"]["Subcelltype"],N=1000,both=True)
            # fig2 = self.plot_umap_fig(x_gen_hat_2,x2_,self.test_set["x2"]["Celltype"],self.test_set["x2"]["Subcelltype"],N=1000,both=True)  
            # fig11 = self.plot_umap_fig(x_gen_hat_1,None,self.test_set["x1"]["Celltype"],self.test_set["x1"]["Subcelltype"],N=1000,both=False)
            # fig22 = self.plot_umap_fig(x_gen_hat_2,None,self.test_set["x2"]["Celltype"],self.test_set["x2"]["Subcelltype"],N=1000,both=False) 
                                    
                                    # self.logger.experiment.add_figure("cond_x1_config_{}/CTM{}/".format(label_config,guidance), fig1, global_step=self.global_step)
                                    # self.logger.experiment.add_figure("cond_x2_config_{}/CTM{}/".format(label_config,guidance), fig2, global_step=self.global_step)   
                                    # self.logger.experiment.add_figure("marg_x1_config_{}/CTM{}/".format(label_config,guidance), fig11, global_step=self.global_step)
                                    # self.logger.experiment.add_figure("marg_x2_config_{}/CTM{}/".format(label_config,guidance), fig22, global_step=self.global_step)     





    def eval_coupling(self,guidance,nb_step,eta=1.0,clip_gen=True):
            self.model_1.eval()
            self.model_2.eval()

            x1_,x2_ =self.data
            x1 = self.preprocess(x1_,0).to(self.device)
            x2 = self.preprocess(x2_,1).to(self.device)

            x_T_1 = torch.randn(*x1.shape,device = self.device)
            x_T_2 = torch.randn(*x2.shape,device = self.device)
            with torch.no_grad(): 
                x_gen_hat_1 = self.sde.sample_from_model(
                 score_function=self.denoiser_1,
                x0=x_T_1, 
                 cond=x2,guidance=guidance,eta=eta,
                clip_gen= clip_gen,
                intervals=self.intervals_[0],
                num_steps=nb_step,ema=False) 
                                    
                x_gen_hat_2 = self.sde.sample_from_model(score_function=self.denoiser_2,
                                                                x0=x_T_2,eta=eta,
                                                                 cond=x1,guidance=guidance,     
                                                                clip_gen= clip_gen,
                                                            intervals=self.intervals_[1],
                                                                num_steps=nb_step,ema=False) 

                       
            x_gen_hat_1 = self.postporcess(x_gen_hat_1,mod =0)
            x_gen_hat_2 = self.postporcess(x_gen_hat_2,mod =1)

        

            x_gen_hat_1_proj = self.find_nearest_neighbors(x_gen_hat_1,x1_)
            x_gen_hat_2_proj = self.find_nearest_neighbors(x_gen_hat_2,x2_)
            x_gen_hat_1 = x_gen_hat_1.cpu().numpy()
            x_gen_hat_2 = x_gen_hat_2.cpu().numpy()
            x2_ = x2_.cpu().numpy() 
            x1_ = x1_.cpu().numpy()
            x_gen_hat_1_proj = x_gen_hat_1_proj.cpu().numpy()
            x_gen_hat_2_proj = x_gen_hat_2_proj.cpu().numpy()

            f1=evals.calc_domainAveraged_FOSCTTM(x_gen_hat_1,x1_)
            f2=evals.calc_domainAveraged_FOSCTTM(x_gen_hat_2,x2_)

            f1_proj=evals.calc_domainAveraged_FOSCTTM(x_gen_hat_1_proj,x1_)
            f2_proj=evals.calc_domainAveraged_FOSCTTM(x_gen_hat_2_proj,x2_)
            n = 4 if self.exp =="snare" else 5
            
            accuracy_1 = evals.transfer_accuracy(x_gen_hat_1,x1_,self.test_set[0] ,self.test_set[1],n=n)
            accuracy_2 = evals.transfer_accuracy(x_gen_hat_2,x2_,self.test_set[1] ,self.test_set[0],n=n)
            accuracy_1_proj = evals.transfer_accuracy(x_gen_hat_1_proj,x1_,self.test_set[0] ,self.test_set[1],n=n)
            accuracy_2_proj = evals.transfer_accuracy(x_gen_hat_2_proj,x2_,self.test_set[1] ,self.test_set[0],n=n)

            return{

                "FOSCTTM":{
                    "x1":np.mean( f1 ),
                    "x2":np.mean( f2 )
                },
                "FOSCTTM_proj":{
                    "x1":np.mean( f1_proj ),
                    "x2":np.mean( f2_proj )
                },
                 "Accuracy":{
                    "x1":accuracy_1,
                    "x2":accuracy_2
                },
                "Accuracy_proj":{
                    "x1":accuracy_1_proj,
                    "x2":accuracy_2_proj
                },
            }, {
                "proj_x": self.plot_pca(x1_,x_gen_hat_1,way="1"),
                "proj_y": self.plot_pca(x2_,x_gen_hat_2,way="2"),
                "proj_x_neighbor":self.plot_pca(x1_,x_gen_hat_1_proj,way="2"),
                "proj_y_neighbor":self.plot_pca(x2_,x_gen_hat_2_proj,way="2")

            }

    @torch.no_grad()
    def log_step_2(self):
            self.model_1.eval()
            self.model_2.eval()
            x1,x2 =self.data
            x1 = self.preprocess(x1,0).to(self.device)
            x2 = self.preprocess(x2,1).to(self.device)
            # x1 = self.preprocess(self.test_set["x1"]["data"]  ,0).to(self.device)
            # x2 = self.preprocess(self.test_set["x2"]["data"]  ,1).to(self.device)
            x1_Celltype = self.test_set["x1"]["Celltype"]
            x1_Subcelltype = self.test_set["x1"]["Subcelltype"]
            x2_Celltype = self.test_set["x2"]["Celltype"]
            x2_Subcelltype = self.test_set["x2"]["Subcelltype"]
  
            x_T_1 = torch.randn(*x1.shape,device = self.device)
            x_T_2 = torch.randn(*x2.shape,device = self.device)
            eta=1.0
            if self.args.ema:
                opt = [True,False]
            else:
                opt = [False]
            for steps in [self.args.num_steps_train]:
                for clip_gen in [False]:
                    for ema in opt:
                        for guidance in [self.args.guidance,3]:
                            label_config = "clip_gen_{}_steps_{}_ema_{}".format(clip_gen,steps,ema)
                            with torch.no_grad(): 
                                    x_gen_hat_1 = self.sde.sample_from_model(
                                                                score_function=self.denoiser_1,
                                                                x0=x_T_1, 
                                                                 c=x2,guidance=self.args.guidance,eta=eta,
                                                                clip_gen= clip_gen,
                                                            intervals=self.intervals_[0],
                                                                num_steps=steps,ema=ema) 
                                    
                                    x_gen_hat_2 = self.sde.sample_from_model(score_function=self.denoiser_2,
                                                                x0=x_T_2,eta=eta,
                                                                 c=x1,guidance=self.args.guidance,     
                                                                clip_gen= clip_gen,
                                                            intervals=self.intervals_[1],
                                                                num_steps=steps,ema=ema) 

                                    x_gen_hat_1 = self.postporcess(x_gen_hat_1,mod =0)
                                    x_gen_hat_2 = self.postporcess(x_gen_hat_2,mod =1)

                                    x1_ = self.postporcess(x1,mod =0)
                                    x2_ = self.postporcess(x2,mod =1)

                                    x_gen_hat_1_proj = self.find_nearest_neighbors(x_gen_hat_1,x1_)
                                    x_gen_hat_2_proj = self.find_nearest_neighbors(x_gen_hat_2,x2_)

                                    x_gen_hat_1 = x_gen_hat_1.cpu().numpy()
                                    x_gen_hat_2 = x_gen_hat_2.cpu().numpy()
                                    x1_ = x1_.cpu().numpy()
                                    x2_ = x2_.cpu().numpy() 
                                    

                                    x_gen_hat_1_proj = x_gen_hat_1_proj.cpu().numpy()
                                    x_gen_hat_2_proj = x_gen_hat_2_proj.cpu().numpy()

                                    CTM_1 = evaluate_neighbors(x_gen_hat_1, x1_, x2_Celltype, x1_Celltype, k=5)
                                    SCTM_1 = evaluate_neighbors(x_gen_hat_1, x1_, x2_Subcelltype, x1_Subcelltype, k=5)

                                    CTM_2 = evaluate_neighbors(x_gen_hat_2, x2_, x1_Celltype, x2_Celltype, k=5)
                                    SCTM_2 = evaluate_neighbors(x_gen_hat_2, x2_, x1_Subcelltype, x2_Subcelltype, k=5)

                                    CTM_1_proj = evaluate_neighbors(x_gen_hat_1_proj, x1_, x2_Celltype, x1_Celltype, k=5)
                                    SCTM_1_proj  = evaluate_neighbors(x_gen_hat_1_proj, x1_, x2_Subcelltype, x1_Subcelltype, k=5)

                                    CTM_2_proj  = evaluate_neighbors(x_gen_hat_2_proj, x2_, x1_Celltype, x2_Celltype, k=5)
                                    SCTM_2_proj  = evaluate_neighbors(x_gen_hat_2_proj, x2_, x1_Subcelltype, x2_Subcelltype, k=5)

                                    print("gen_step",self.gen_step,label_config,guidance,"CTM:", CTM_1, CTM_2)
                                    print("gen_step",self.gen_step,label_config,guidance,"SCTM:",SCTM_1, SCTM_2)

                                    print("gen_step",self.gen_step,label_config,guidance,"CTM:", CTM_1_proj, CTM_2_proj)
                                    print("gen_step",self.gen_step,label_config,guidance,"SCTM:",SCTM_1_proj, SCTM_2_proj)
                                    
                                    self.logger.experiment.add_scalars("config_{}/CTM{}/".format(label_config,guidance), 
                                                                                {"x1": CTM_1,
                                                                                "x2":CTM_2 },self.gen_step )
                                    self.logger.experiment.add_scalars("config_{}/SCTM{}/".format(label_config,guidance), 
                                                                                {"x1": SCTM_1,
                                                                                "x2":SCTM_2 },self.gen_step )

                                    self.logger.experiment.add_scalars("config_{}_proj/CTM{}/".format(label_config,guidance), 
                                                                                {"x1": CTM_1_proj,
                                                                                "x2":CTM_2_proj },self.gen_step )
                                    self.logger.experiment.add_scalars("config_{}_proj/SCTM{}/".format(label_config,guidance), 
                                                                                {"x1": SCTM_1_proj,
                                                                                "x2":SCTM_2_proj },self.gen_step )

                                    # fig1 = self.plot_umap_fig(x_gen_hat_1,x1_,self.test_set["x2"]["Celltype"],self.test_set["x1"]["Subcelltype"],N=500,both=True)
                                    # fig2 = self.plot_umap_fig(x_gen_hat_2,x2_,self.test_set["x1"]["Celltype"],self.test_set["x2"]["Subcelltype"],N=500,both=True)  

                                    # fig11 = self.plot_umap_fig(x_gen_hat_1,None,self.test_set["x2"]["Celltype"],self.test_set["x2"]["Subcelltype"],N=500,both=False)
                                    # fig22 = self.plot_umap_fig(x_gen_hat_2,None,self.test_set["x1"]["Celltype"],self.test_set["x1"]["Subcelltype"],N=500,both=False) 
                                    
                                    # self.logger.experiment.add_figure("cond_x1_config_{}/CTM{}/".format(label_config,guidance), fig1, global_step=self.global_step)
                                    # self.logger.experiment.add_figure("cond_x2_config_{}/CTM{}/".format(label_config,guidance), fig2, global_step=self.global_step)   
                                    # self.logger.experiment.add_figure("marg_x1_config_{}/CTM{}/".format(label_config,guidance), fig11, global_step=self.global_step)
                                    # self.logger.experiment.add_figure("marg_x2_config_{}/CTM{}/".format(label_config,guidance), fig22, global_step=self.global_step)     


    @torch.no_grad()
    def log_step_1(self):
            self.model_1.eval()
            self.model_2.eval()
            x1,x2 =self.data
            x1 = self.preprocess(x1,0).to(self.device)
            x2 = self.preprocess(x2,1).to(self.device)
            x_T_1 = torch.randn(*x1.shape,device = self.device)
            x_T_2 = torch.randn(*x2.shape,device = self.device)
            eta =1.0

            if self.args.ema:
                opt = [True,False]
            else:
                opt = [False]
            steps = self.args.num_steps_train
            for ema in opt:
                for clip_gen in [False]:
                    for eta in [1]:
                        for guidance in [3,self.args.guidance]:
                            label_config = "guidance_{}_clip_gen_{}_steps_{}_ema_{}_eta_{}".format(guidance,clip_gen,self.args.num_steps_train,ema,eta)
                            with torch.no_grad(): 
                                    x_gen_hat_1 = self.sde.sample_from_model(
                                                                score_function=self.denoiser_1,
                                                                x0=x_T_1, 
                                                                 c=x2,guidance=guidance,eta=eta,
                                                                clip_gen= clip_gen,
                                                            intervals=self.intervals_[0],
                                                                num_steps=steps,ema=ema) 
                                    x_gen_hat_2 = self.sde.sample_from_model(score_function=self.denoiser_2,
                                                                x0=x_T_2,eta=eta,
                                                                 c=x1,guidance=guidance,     
                                                                clip_gen= clip_gen,
                                                            intervals=self.intervals_[1],
                                                                num_steps=steps,ema=ema) 

                                    if True:
                                        x_gen_hat_1 = self.postporcess(x_gen_hat_1,mod =0)
                                        x_gen_hat_2 = self.postporcess(x_gen_hat_2,mod =1)

                                        x1_ = self.postporcess(x1,mod =0)
                                        x2_ = self.postporcess(x2,mod =1)

                                        x_gen_hat_1_proj = self.find_nearest_neighbors(x_gen_hat_1,x1_)
                                        x_gen_hat_2_proj = self.find_nearest_neighbors(x_gen_hat_2,x2_)
                                        x_gen_hat_1 = x_gen_hat_1.cpu().numpy()
                                        x_gen_hat_2 = x_gen_hat_2.cpu().numpy()
                                        x2_ = x2_.cpu().numpy() 
                                        x1_ = x1_.cpu().numpy()
                                        x_gen_hat_1_proj = x_gen_hat_1_proj.cpu().numpy()
                                        x_gen_hat_2_proj = x_gen_hat_2_proj.cpu().numpy()

                                        

                                        f1=evals.calc_domainAveraged_FOSCTTM(x_gen_hat_1,x1_)
                                        f2=evals.calc_domainAveraged_FOSCTTM(x_gen_hat_2,x2_)

                                        f1_proj=evals.calc_domainAveraged_FOSCTTM(x_gen_hat_1_proj,x1_)
                                        f2_proj=evals.calc_domainAveraged_FOSCTTM(x_gen_hat_2_proj,x2_)
                                        n = 4 if self.exp =="snare" else 5
                                        accuracy_1 = evals.transfer_accuracy(x_gen_hat_1,x1_,self.test_set[0] ,self.test_set[1],n=n)
                                        accuracy_2 = evals.transfer_accuracy(x_gen_hat_2,x2_,self.test_set[1] ,self.test_set[0],n=n)
                                        accuracy_1_proj = evals.transfer_accuracy(x_gen_hat_1_proj,x1_,self.test_set[0] ,self.test_set[1],n=n)
                                        accuracy_2_proj = evals.transfer_accuracy(x_gen_hat_2_proj,x2_,self.test_set[1] ,self.test_set[0],n=n)

                                        print("gen_step",self.gen_step,label_config,guidance,"FOSCTTM:", np.mean( f1 ),np.mean( f2 ))
                                        print("gen_step",self.gen_step,label_config,guidance,"Accuracy:",accuracy_1,accuracy_2)
                                        print("gen_step",self.gen_step,label_config,guidance,"FOSCTTM_proj:", np.mean( f1_proj ),np.mean( f2_proj ))
                                        print("gen_step",self.gen_step,label_config,guidance,"Accuracy_proj:",accuracy_1_proj,accuracy_2_proj)

                                        self.logger.experiment.add_scalars("rl_{}/FOSCTTM/".format(label_config), 
                                                                                {"x1": np.mean( f1 ),
                                                                                "x2":np.mean( f2 ) ,
                                                                                "gt":0},self.gen_step )
                                        self.logger.experiment.add_scalars("rl_{}/FOSCTTM_proj/".format(label_config), 
                                                                                {"x1": np.mean( f1_proj ),
                                                                                "x2":np.mean( f2_proj ) ,
                                                                                "gt":0},self.gen_step )

                                        self.logger.experiment.add_scalars("rl_{}/accuracy/".format(label_config), 
                                                                                {"x1": accuracy_1,
                                                                                "x2":accuracy_2 },self.gen_step )
                                        self.logger.experiment.add_scalars("rl_{}/accuracy_proj/".format(label_config), 
                                                                                {"x1": accuracy_1_proj,
                                                                                "x2":accuracy_2_proj },self.gen_step )                                       

                                        self.logger.experiment.add_figure("rl_{}/x1/".format(label_config), self.plot_pca(x1_,x_gen_hat_1,way="1"), global_step=self.global_step)
                                        self.logger.experiment.add_figure("rl_{}/x2/".format(label_config), self.plot_pca(x2_,x_gen_hat_2,way="2"), global_step=self.global_step)
                                        self.logger.experiment.add_figure("rl_{}/x1_marg/".format(label_config), self.plot_pca_alone(x_gen_hat_1,way="1"), global_step=self.global_step)
                                        self.logger.experiment.add_figure("rl_{}/x2_marg/".format(label_config), self.plot_pca_alone(x_gen_hat_2,way="2"), global_step=self.global_step)    
                                        # self.logger.experiment.add_figure("rl/data/x1/", self.plot_pca_alone(x1_,way="1"), global_step=self.global_step)
                                        # self.logger.experiment.add_figure("rl/data/x2", self.plot_pca_alone(x2_,way="2"), global_step=self.global_step)    



    @torch.no_grad()
    def log_step_warmup_1(self):
            self.model_1.eval()
            self.model_2.eval()
            x1_,x2_ =self.data
            x1 = self.preprocess(x1_,0).to(self.device)
            x2 = self.preprocess(x2_,1).to(self.device)
           # x_T_1 = torch.randn(*x1.shape,device = self.device)
           # x_T_2 = torch.randn(*x2.shape,device = self.device)

            x_T_1 = torch.randn(*x2.shape,device = self.device)
            x_T_2 = torch.randn(*x1.shape,device = self.device)
            if self.args.ema:
                opt = [True,False]
            else:
                opt = [False]
            steps = self.args.num_steps_train
            clip_gen = self.args.clip_gen 
            with torch.no_grad():
                for ema in opt:
                    for guidance in [0,self.args.guidance]:
                        x_gen_hat_1 = self.sde.sample_from_model(
                                                            score_function=self.denoiser_1,
                                                            x0=x_T_1, 
                                                            c=x2,guidance=guidance,eta=1.0,
                                                            clip_gen= clip_gen,
                                                        intervals=self.intervals_[0],
                                                            num_steps=self.args.num_steps_train,ema=ema) 
                                
                        x_gen_hat_2 = self.sde.sample_from_model(score_function=self.denoiser_2,
                                                            x0=x_T_2,eta=1.0,
                                                            c=x1,guidance=guidance, 
                                                            clip_gen=  clip_gen,
                                                        intervals=self.intervals_[1],
                                                            num_steps=self.args.num_steps_train,ema=ema) 
                    
                        x_gen_hat_1 = self.postporcess( x_gen_hat_1,mod =0 ).cpu().numpy()
                        x_gen_hat_2 = self.postporcess(x_gen_hat_2,mod =1).cpu().numpy()


                        # x1_ = x1_.cpu().numpy()
                        # x2_ = x2.cpu().numpy() 
                                                
                         
                        self.logger.experiment.add_figure("warmup/x1_marg/guidance"+str(guidance)+"_ema"+str(ema)+"/", self.plot_pca_alone(x_gen_hat_1,way="1"), global_step=self.global_step)
                        self.logger.experiment.add_figure("warmup/x2_marg/guidance"+str(guidance)+"_ema"+str(ema)+"/", self.plot_pca_alone(x_gen_hat_2,way="2"), global_step=self.global_step)    
                        self.logger.experiment.add_figure("warmup/x1_data/", self.plot_pca_alone(x1_,way="1"), global_step=self.global_step)
                        self.logger.experiment.add_figure("warmup/x2_data/", self.plot_pca_alone(x2_,way="2"), global_step=self.global_step)  


    def plot_umap_fig(self,source_projected, target_projected, cell_type , sub_cell_type ,N = 1000 ,both = None): 
        fig, ax = plt.subplots(figsize=(4, 3))
        source_projected = source_projected[:N]
        cell_type= cell_type[:N]
        sub_cell_type = sub_cell_type[:N]

        if   both:
            target_projected= target_projected[:N]
            batch = np.concatenate((np.repeat('ATAC',source_projected.shape[0]),np.repeat('RNA',target_projected.shape[0])))

            celltype = np.concatenate((cell_type,cell_type))
            subcelltype = np.concatenate((sub_cell_type,sub_cell_type))

            Aligned_metadata = pd.DataFrame(np.array([batch,celltype,subcelltype]).T,columns=['Batch','Celltype','Subcelltype'],
                                            )
            Aligned_data = pd.DataFrame(np.concatenate((source_projected,target_projected),axis=0))

            adata_RNA_ATAC_aligned = sc.AnnData(X = Aligned_data, obs = Aligned_metadata)
            sc.pp.neighbors(adata_RNA_ATAC_aligned, n_neighbors=30, n_pcs=0)
            sc.tl.umap(adata_RNA_ATAC_aligned)
            fig = sc.pl.umap(adata_RNA_ATAC_aligned, color=['Batch',"Celltype"], title = 'Dataset',return_fig=True)
        else:
            batch = np.repeat('ATAC',source_projected.shape[0])

            celltype =cell_type  
            subcelltype = sub_cell_type #np.concatenate((sub_cell_type,sub_cell_type))

            Aligned_metadata = pd.DataFrame(np.array([batch,celltype,subcelltype]).T,columns=['Batch','Celltype','Subcelltype'])
            Aligned_data = pd.DataFrame(np.concatenate((source_projected,),axis=0))
            adata_RNA_ATAC_aligned = sc.AnnData(X = Aligned_data, obs = Aligned_metadata)
            sc.pp.neighbors(adata_RNA_ATAC_aligned, n_neighbors=30, n_pcs=0)
            sc.tl.umap(adata_RNA_ATAC_aligned)
            fig = sc.pl.umap(adata_RNA_ATAC_aligned, color=["Celltype"], title = 'Dataset',return_fig=True)
        
        return fig


    @torch.no_grad()
    def log_step_warmup_2(self):
            self.model_1.eval()
            self.model_2.eval()
            x1,x2 =self.data
            x_T_1 = torch.randn(*x1.shape,device = self.device)
            x_T_2 = torch.randn(*x2.shape,device = self.device)
            if self.args.ema:
                opt = [True,False]
            else:
                opt = [False]
            with torch.no_grad(): 
                for ema in opt:
                    x_gen_hat_1 = self.sde.sample_from_model(
                                                        score_function=self.denoiser_1,
                                                        x0=x_T_1, 
                                                         c=None,guidance=0,eta=1.0,
                                                        clip_gen= self.args.clip_gen if self.intervals_[0] !=None else False,
                                                    intervals=self.intervals_[0],
                                                        num_steps=self.args.num_steps_train,ema=ema) 
                            
                    x_gen_hat_2 = self.sde.sample_from_model(score_function=self.denoiser_2,
                                                        x0=x_T_2,eta=1.0,
                                                         c=None,guidance=0, 
                                                        clip_gen= self.args.clip_gen if self.intervals_[0] !=None else False,
                                                    intervals=self.intervals_[1],
                                                        num_steps=self.args.num_steps_train,ema=ema) 

                    x_gen_hat_1 = self.postporcess( x_gen_hat_1,mod =0 ).cpu().numpy()
                    x_gen_hat_2 = self.postporcess(x_gen_hat_2,mod =1).cpu().numpy()


                    x1_ = x1.cpu().numpy()
                    x2_ = x2.cpu().numpy() 

                    fig1 = self.plot_umap_fig(x_gen_hat_1,None,self.test_set["x1"]["Celltype"],self.test_set["x1"]["Subcelltype"],N=1000,both=False)
                    fig2 = self.plot_umap_fig(x_gen_hat_2,None,self.test_set["x2"]["Celltype"],self.test_set["x2"]["Subcelltype"],N=1000,both=False)  

                    fig11 = self.plot_umap_fig(x1_,None,self.test_set["x1"]["Celltype"],self.test_set["x1"]["Subcelltype"],N=1000,both=False)
                    fig22 = self.plot_umap_fig(x2_,None,self.test_set["x2"]["Celltype"],self.test_set["x2"]["Subcelltype"],N=1000,both=False) 

                    self.logger.experiment.add_figure("x1/marg"+str(ema)+"/", fig1, global_step=self.global_step)
                    self.logger.experiment.add_figure("x2/marg"+str(ema)+"/", fig2, global_step=self.global_step)    
                    self.logger.experiment.add_figure("x1/data/", fig11, global_step=self.global_step)
                    self.logger.experiment.add_figure("x2/data/", fig22, global_step=self.global_step)  

    def plot_pca_alone(self, X_origin, way="1"):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import numpy as np

        # X_origin =  X_origin.cpu().numpy()
        # X_trans =X_trans.cpu().numpy()

        # Load cell type data (ensure these are loaded correctly)
       # cellTypes_atac = np.loadtxt("SCOT/data/SNARE/SNAREseq_atac_types.txt")
       # cellTypes_rna = np.loadtxt("SCOT/data/SNARE/SNAREseq_rna_types.txt")
        
        # cellTypes_atac = pd.read_csv('data/cellid_early.csv', header=0)
        # cellTypes_rna = pd.read_csv('data/cellid_late.csv', header=0)
        # cellTypes_atac = cellTypes_atac.iloc[:,0].to_numpy()
        # cellTypes_rna = cellTypes_rna.iloc[:,0].to_numpy()
        
        cellTypes_atac =self.test_set[0]
        cellTypes_rna = self.test_set[1]
        unique_labels_1, cellTypes_atac = np.unique(cellTypes_atac, return_inverse=True)
        unique_labels_2, cellTypes_rna = np.unique(cellTypes_rna, return_inverse=True)

        # Decide which cell types to use based on the 'way' argument
        if way == "1":
            cell_types_1 = cellTypes_atac
            cell_types_2 = cellTypes_rna
            tick = unique_labels_1
        else:
            cell_types_1 = cellTypes_rna
            cell_types_2 = cellTypes_atac   
            tick = unique_labels_2

        # Ensure X_new and y_new have compatible dimensions
        N = X_origin.shape[0]  # Number of samples
      

        # Reduce the dimensionality of the aligned domains to two (2D) via PCA:
        pca = PCA(n_components=2)

        Xy_pca = pca.fit_transform(X_origin)

        X_pca = Xy_pca[:N, :]
   

        # Plot aligned domains, samples colored by domain identity:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define the colormap for coloring based on cell types
        colormap = plt.get_cmap('rainbow', 4)  # Adjust '4' if you have more than 4 unique classes
        
        # Plot the first domain (X_pca)
        scatter1 = ax.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', s=30, c=cell_types_1, label="Gene Expression", cmap=colormap)

        cbar = plt.colorbar(scatter1)  # Colorbar based on first scatter
        tick_locs = (np.arange(1, 5) + 0.75) * 3 / len(tick)  # Adjust tick locations as needed
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick)  # Change labels as per your data

        # Add plot title and legend
        ax.legend()
        ax.set_title("Colored Based on Domains (Aligned Domains)")
        
        # Return the figure
        return fig  # Return the matplotlib figure



    def plot_marginals_comparison(self,marginals, predicted_marginals, title="Comparison of Marginals and Predicted Marginals"):
        import matplotlib.pyplot as plt
        marginals = marginals.cpu().numpy().reshape(-1)
        predicted_marginals = predicted_marginals.cpu().numpy().reshape(-1)
      

        fig, ax = plt.subplots()
        width = 0.35  # the width of the bars

        indices = np.arange(len(marginals))
        ax.bar(indices - width/2, marginals, width, label='Marginals')
        ax.bar(indices + width/2, predicted_marginals, width, label='Predicted Marginals')

        ax.set_xlabel('Element')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        return fig  # Return the matplotlib figure


    def postporcess(self,x,mod,proj=False):
            x_min = self.intervals[mod][0]
            x_max = self.intervals[mod][1]
            x = x.clamp(x_min,x_max)

            if self.preporcess_ =="min_max":
                x_min = self.intervals[mod][0]
                x_max = self.intervals[mod][1]
                x = x.clamp(-1,1)
                x =  ((x + 1) / 2) * (x_max - x_min) + x_min
            elif self.preporcess_ == None:
                x =  x
            else:
                x=  (x * self.stat[mod]["std"].to(self.device)) + self.stat[mod]["mean"].to(self.device) 
                x = x.clamp(self.stat[mod]["min"].to(self.device) , self.stat[mod]["max"].to(self.device))
            if proj :
                return self.find_nearest_neighbors(x, self.data[mod])
            else:
                return x

    def preprocess(self,x,mod):
        if self.preporcess_ =="min_max":
            x_min = self.intervals[mod][0]
            x_max = self.intervals[mod][1]
            return 2 * (x - x_min) / (x_max - x_min) - 1
        elif self.preporcess_ == None:
                return x
        else:
            return (x - self.stat[mod]["mean"].to(x.device)) / self.stat[mod]["std"].to(x.device)   



    def plot_pca(self, X_origin, X_trans, way="1"):
        """
        Plots PCA-transformed representations of two domains (X_origin and X_trans),
        colored based on domain identity.
        
        Parameters:
        - X_origin: Original feature matrix (numpy array)
        - X_trans: Transformed feature matrix (numpy array)
        - way: String indicating mapping direction ("1" or otherwise)
        
        Returns:
        - A matplotlib figure displaying the PCA projection.
        """
        # Determine dataset-specific labels
        if way == "1":
            cell_types_1, cell_types_2 = self.test_set[0], self.test_set[1]
            labels = ["Chromatin Accessibility", "Gene Expression"] if self.exp == "snare" \
                    else ["Gene Expression", "DNA Methylation"]
        else:
            cell_types_1, cell_types_2 = self.test_set[1], self.test_set[0]
            labels = ["Gene Expression", "Chromatin Accessibility"] if self.exp == "snare" \
                    else ["DNA Methylation", "Gene Expression"]
        
        # Number of samples
        N = X_origin.shape[0]
        
        # Apply PCA for dimensionality reduction (to 2D)
        pca = PCA(n_components=2)
        Xy_pca = pca.fit_transform(np.vstack((X_origin, X_trans)))
        X_pca, y_pca = Xy_pca[:N, :], Xy_pca[N:, :]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 4))
        colormap = plt.get_cmap('RdYlBu', 4)  # Adjust if more than 4 unique classes
        
        # Scatter plots for both domains
        scatter1 = ax.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', s=30, c=cell_types_1, cmap=colormap, label="GT", alpha=1.0, edgecolors='gray')
        scatter2 = ax.scatter(y_pca[:, 0], y_pca[:, 1], marker='^', s=50, c=cell_types_2, cmap=colormap, label="Generated", alpha=1.0, edgecolors='gray')
        
        # Add color bar
        cbar = plt.colorbar(scatter1)
        tick_locs = (np.arange(1, 5) + 0.75) * 3 / 4  # Adjust as needed
        cbar_labels = ["H1", "GM", "BJ", "K562"] if self.exp == "snare" else ["BJ", "d8", "d16T+", "d24T+", "iPS"]
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(cbar_labels)
        
        # Formatting
        ax.legend()
      #  ax.set_title("PCA Projection Colored by Domain Identity")
        
        return fig

   


    def configure_optimizers(self):
        opt1 = torch.optim.AdamW(self.model_1.parameters(), lr=self.args.lr[0])
        opt2 = torch.optim.AdamW(self.model_2.parameters(), lr=self.args.lr[0])
        return opt1,opt2 
    
   

            
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return 
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        return



   