import numpy as np
import torch
from torchdiffeq import odeint
from util.util import reshape_for_batchwise, plot_loss_curve, plot_real_vs_fake
import time
from util.util import save_checkpoint, fmt

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



import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append('../')
 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torchcfm.optimal_transport import OTPlanSampler

from util.gaussian_process import *
from functional_fm import *
from models.models_fno.fno import FNO 
from util.util import reshape_for_batchwise
from util.ema import EMA
from util.util import load_checkpoint


from kl_functions_torch import project_to_basis, reconstruct_from_basis, integration_t # *
# basis = 'fourier'

 
def check_error( 
                Xt_hat_input,   vt_hat_output, 

                mean_hat_train, 

                lam_k_C, lam_k_K, t, 

                mean_dims = (0,),
                ):


    ###########################################
    # check 1 : compare with analytic vt_hat
        
    vt_hat_analytic = get_vk_vdiffk_array( 
        m1=mean_hat_train, # torch.zeros_like(mean_hat), 
        c=lam_k_C, k=lam_k_K, t=t, 
        r = Xt_hat_input, return_item = 'v'
        )   

    vt_hat_error = (torch.abs(vt_hat_output - vt_hat_analytic)**2).mean(dim=mean_dims).sum() # .mean(dim=0).sum()

    return vt_hat_error, vt_hat_analytic


 
def kl_gt_compute(mean_hat_A, lam_k_C): 
    KL_gt_p1 = kl_gaussian_mean_highD_GT(mean_hat_A[0], lam_k_C) # 0.5 * np.sum( np.real( np.conj(m1k_array) * m1k_array ) / sk_array_C )
    KL_gt_p2 = kl_gaussian_mean_highD_GT(mean_hat_A[1], lam_k_C) # 0.5 * np.sum( np.real( np.conj(m1k_array) * m1k_array ) / sk_array_C )
    KL_gt = 0.5 * KL_gt_p1 + 0.5 * KL_gt_p2 
    return KL_gt 



class FKLModel:
    def __init__(self, 
                ffm_model_A, ffm_model_B, 
                ):
        self.ffm_model_A = ffm_model_A
        self.ffm_model_B = ffm_model_B

        self.ffm_model_A.model.eval()
        self.ffm_model_B.model.eval()


        self.device = ffm_model_A.device

        self.ot_sampler = OTPlanSampler(method="exact")
    

    def print_rows(self, dictionary):
        row = "\t".join(fmt(v) for v in dictionary.keys()) 
        print(row)
        row = "\t".join(fmt(v) for v in dictionary.values()) 
        print(row)


    

    def t_list_compute(self, n_t=100, t_kl_sampling_scheme='uniform'):

        print(f"KL: n_t={n_t}, t_kl_sampling_scheme={t_kl_sampling_scheme}")
        if t_kl_sampling_scheme == "uniform":
            t_list = torch.linspace(
               0.01,  0.99,  
                            #    0.01,  0.999,  
                                # 0.001,  0.999,  
                                #   0.01,  0.95,  

                                # 0.001, 0.999,
                               
                                    n_t, device=self.device)  
           
            sum_way = 'riemann'  
            return t_list, sum_way
       
        
        elif t_kl_sampling_scheme == "importance_sampling_t/(1-t)":
            

            # ---- Integration over FFM's t ----
            t_eps =  0.96 #   0.96 # 7 # 0.996
            print(t_kl_sampling_scheme, 't_eps', t_eps)


            rng = torch.Generator(device=self.device)
            rng.manual_seed(int(3))
            # rng = None

            t_list, normalizing_constant  = sample_trunc_t_over_1mt(t_eps, n_t, return_Z=True,   rng=rng, sort=True,)
            sum_way = 'uniform'
            # print(t_list.detach().cpu().numpy())

            return t_list, sum_way, normalizing_constant
            

        elif t_kl_sampling_scheme == "importance_sampling_t*(1-t)":
            t_list = torch.distributions.Beta(2.0, 2.0).sample((n_t,)).to(device)
            t_list = torch.unique(t_list.clamp(max=0.9))   # or: torch.clamp(t_list, max=0.9)


            orig_shape = t_list.shape
            t_sorted, _ = torch.sort(t_list.reshape(-1))  # GPU sort
            t_list = t_sorted.reshape(orig_shape)

            normalizing_constant = 1.0 / 6.0
            sum_way = 'uniform'
            # print(t_list.detach().cpu().numpy())

            return t_list, sum_way, normalizing_constant
            
        
        else:
            raise ValueError(f"Unknown  t_kl_sampling_scheme: { t_kl_sampling_scheme}")

            
                
            

    
    

    @torch.no_grad()
    def kl_estimate(self, 
                    X1_data_A, 
                    
                    # ---- stats ----
                    upsample,
                    Phi,
                    lam_k_K,

                    analytic_stats=None, # mean_hat_A, mean_hat_B, lam_k_C = analytic_stats
                                        
                    # ---- Integration over FFM's t ----
                    n_t = 100,  
                    t_kl_sampling_scheme = False,  
                    # ---- (diff of) v_t 's source ----
                    # way 1
                    vdiff_source = 'sample',
                    v_source = 'neural_network',

                  

                    sfolder = '',

                    X0_times = 1,
 

                    ):
        
       
        print(f"loop noise sampling for {X0_times} times")


        ############################################################
        print(f'to compute KL at upsample={upsample}')
        n_kl_samples = X1_data_A.shape[0]


        if v_source == 'neural_network':
            print("check boundary v_1(x) at t=1: =x?")
            Xt_data_A = X1_data_A
            t_inp     = torch.tensor([1.0 - 0.05],   device=device).expand(n_kl_samples)

            vt_data_A = self.ffm_model_A.model(t_inp, Xt_data_A.transpose(-1, -2)).transpose(-1, -2)
            vt_data_B = self.ffm_model_B.model(t_inp, Xt_data_A.transpose(-1, -2)).transpose(-1, -2)

            plot_real_vs_fake(
                            y_real=X1_data_A, 
                            y_fake=vt_data_A, 
                            save_path=Path(f'../{sfolder}/v1A(x1A)_vs_x1A.pdf')
                            )
            plot_real_vs_fake(
                            y_real=X1_data_A, 
                            y_fake=vt_data_B, 
                            save_path=Path(f'../{sfolder}/v1B(x1A)_vs_x1A.pdf')
                            )
            
        kl_x0_t_log = []
        for X0_time in tqdm(range(X0_times), desc="X0", ncols=80):
            
       
            print(f"sample {n_kl_samples} XO per X0_time")
            X0_hat, X0_data = self.ffm_model_A.gp.sample(n_samples=n_kl_samples,   upsample=upsample,
                        return_hat = True
                        )  

           
            
            lam_k_K = self.ffm_model_A.gp.lam_k
            Phi = self.ffm_model_A.gp.Phi


            M = X1_data_A.shape[1] // upsample
            assert X1_data_A.shape[1] == Phi.shape[0]
            D = X1_data_A.shape[2]
            N = self.ffm_model_A.gp.N # (Phi.shape[1] - 1) // 2
            assert N < int(M*upsample)/2, "Nyquist condition is not satisfied!"  


 

            X1_hat_A = project_to_basis(X1_data_A, Phi)
            
            # assert abs(self.ffm_model_A.gp.Phi - Phi).max() == 0

            assert X1_hat_A.shape == X0_hat.shape


            ############################################################
            if not ( (vdiff_source == 'sample') and (v_source == 'neural_network') ):
                assert analytic_stats is not None
            if analytic_stats is not None:
                mean_hat_A, mean_hat_B, lam_k_C = analytic_stats

  
            # ---- Integration over FFM's t ----
            t_list_returns = self.t_list_compute(n_t=n_t, t_kl_sampling_scheme=t_kl_sampling_scheme)
            if t_kl_sampling_scheme == 'uniform':
                t_list, sum_way = t_list_returns
            else:
                t_list, sum_way, normalizing_constant = t_list_returns


            ###########################################################################
            # ---- Integration over FFM's t ----
            # Loop over t only, thanks to the closed-form expectation wrt Xt


            vt_hat_A_error_list = []
            vt_hat_B_error_list = []
            vt_hat_diff_error_list = []


            kl_result = []
            
            # if vdiff_source == 'sample': # prepare data X1 and noise X0

            # for noise, for t log 
                
            for t in tqdm(t_list, desc="t", ncols=80):

                if vdiff_source == 'sample':
                    Xt_hat_A  = t * X1_hat_A + (1.0 - t) * X0_hat


                    if v_source == 'analytic':
                        vt_hat_A = get_vk_vdiffk_array( 
                            m1=mean_hat_A, 
                            c=lam_k_C, k=lam_k_K, t=t, 
                            r = Xt_hat_A, return_item = 'v'
                            )   
                        vt_hat_B = get_vk_vdiffk_array( 
                            m1=mean_hat_B, # torch.zeros_like(mean_hat_A), 
                            c=lam_k_C, k=lam_k_K, t=t, 
                            r = Xt_hat_A, return_item = 'v'
                            )   
            
                    elif v_source == 'neural_network':
                        Xt_data_A = reconstruct_from_basis(Xt_hat_A, Phi)
                        t_inp     = torch.tensor([t],   device=device).expand(n_kl_samples)
 
                        f =  self.ffm_model_A.vt_from_output( self.ffm_model_A.prediction  )
                        img = Xt_data_A.transpose(-1, -2)
                        vt_data_A = f(t_inp, img, self.ffm_model_A.model).transpose(-1, -2)
                        vt_data_B = f(t_inp, img, self.ffm_model_B.model).transpose(-1, -2)
                        # print(t.item(), torch.abs((vt_data_A - vt_data_B)**2).sum().item())
                    

                        if t_kl_sampling_scheme == "importance_sampling_t*(1-t)" and self.ffm_model_A.prediction == 'vt':
                            # transform vt to mt
                            vt_data_A = (Xt_data_A - vt_data_A) / (1-t)
                            vt_data_B = (Xt_data_A - vt_data_B) / (1-t)


                        vt_hat_A = project_to_basis(vt_data_A, Phi)
                        vt_hat_B = project_to_basis(vt_data_B, Phi)


                        ###########################################
                        if analytic_stats is not None:
                            # check 1 : compare with analytic vt_hat
                            vt_hat_A_error, vt_hat_A_analytic = check_error(
                                                Xt_hat_input=Xt_hat_A,   vt_hat_output=vt_hat_A, 

                                                mean_hat_train=mean_hat_A, 

                                                lam_k_C=lam_k_C, lam_k_K=lam_k_K, t=t, 

                                                mean_dims=(0,),
                                                )     
                            
                            vt_hat_B_error, vt_hat_B_analytic = check_error(
                                                Xt_hat_input=Xt_hat_A,   vt_hat_output=vt_hat_B, 

                                                mean_hat_train=mean_hat_B, 

                                                lam_k_C=lam_k_C, lam_k_K=lam_k_K, t=t, 

                                                mean_dims=(0,),
                                                )     
                                                    
                            vt_hat_A_error_list.append( vt_hat_A_error )
                            vt_hat_B_error_list.append( vt_hat_B_error )

                            vt_hat_diff_error = (torch.abs( (vt_hat_A - vt_hat_B) - (vt_hat_A_analytic-vt_hat_B_analytic) )**2).mean(dim=0).sum() 
                            vt_hat_diff_error_list.append( vt_hat_diff_error )

                                   
                        ###########################################

                    
                    vdiffk_array = vt_hat_A - vt_hat_B

             


                elif vdiff_source == 'analytic':
                    
                    vdiffk_array = get_vk_vdiffk_array( 
                        m1=mean_hat_A - mean_hat_B, c=lam_k_C, k=lam_k_K, t=t, 
                        r = None, return_item = 'vdiff' 
                        )  

                v_diff_2   = torch.abs(vdiffk_array)**2
            
                if self.ffm_model_A.gp.basis == 'fourier':
                    assert v_diff_2.shape == (n_kl_samples, 2*N+1, D) 
                elif self.ffm_model_A.gp.basis == 'cosine':
                    assert v_diff_2.shape == (n_kl_samples, N+1, D)
                # print(t.item(), v_diff_2.sum().item())

                kl_nocoeft = v_diff_2 / lam_k_K[None, :, None]

                # N_use = N  # 16 # 
                # print(N, N_use)
                # kl_nocoeft = kl_nocoeft[:, N-N_use:N+N_use+1, :]
            
                kl_nocoeft = torch.sum( 
                    kl_nocoeft , 
                    dim = -2 
                )
                assert kl_nocoeft.shape == (n_kl_samples, D) 
            
                
                if t_kl_sampling_scheme == 'uniform':
                    if 'vt' in self.ffm_model_A.prediction :
                        coef_t = t / (1. - t)  
                        # print(t.item(), coef_t.item())
                    elif self.ffm_model_A.prediction == 'mt':
                        coef_t = t * (1. - t)  
                else:
                    coef_t = normalizing_constant 
                
                kl_coeft = coef_t * kl_nocoeft 
                kl_coeft = torch.real(kl_coeft) 

                kl_result.append( kl_coeft ) 
        
        
            kl_result = torch.stack(kl_result, dim=1) 
            assert kl_result.shape == (n_kl_samples, len(t_list), D)

            for t, kl_t in zip(t_list, kl_result.mean(dim=0).mean(dim=-1)):
                print(f"t={t:.3f}: kl_t={kl_t.item():.6f}")

            # Sum  over D 
            kl_result = torch.sum(kl_result, dim=-1, keepdim=True)
            assert kl_result.shape == (n_kl_samples, len(t_list), 1)

            kl_x0_t_log.append(kl_result)


        
        kl_x0_t_log = torch.cat(kl_x0_t_log, dim=-1)
        assert kl_x0_t_log.shape == (n_kl_samples, len(t_list), X0_times)

     
        kl_result = torch.mean( kl_x0_t_log , dim=-1, keepdim=True )  # average over X0_time
        assert kl_result.shape == (n_kl_samples, len(t_list), 1)

        # Sum over t
        KL_est = integration_t(kl_result, t_list, sum_way=sum_way)
        assert KL_est.shape == (n_kl_samples, 1)
        KL_est = KL_est.squeeze(-1)
        assert KL_est.shape == (n_kl_samples,)


        


        # Mean over samples
        KL_sample_std =  torch.std(KL_est) 
        KL_sample_mean = torch.mean(KL_est)


        


        # ---- KL comparison ----
        # KL_GT
        if analytic_stats is not None:
            KL_gt = kl_gt_compute(mean_hat_A, lam_k_C)
 


        # KL comparison
        print("\n=== Hyperparameters ===")
        hyper = dict(
            upsample=upsample, 
            n_t=n_t,
            t_kl_sampling_scheme=t_kl_sampling_scheme,
            
        )
        self.print_rows(hyper)

        print("\n=== KL result ===")
        if analytic_stats is not None:
            KL_ratio = (KL_sample_mean / KL_gt).item()
            result_kl = dict(
                KL_GT = KL_gt.item(),
                KL_Est = KL_sample_mean.item(),
                KL_ratio = KL_ratio,
            )
        else:
            result_kl = dict(
                KL_Est = KL_sample_mean.item(), 
            )

        self.print_rows(result_kl)

    

        if (vdiff_source == 'sample') and (v_source == 'neural_network') and (analytic_stats is not None):  
            plt.figure(figsize=(6, 4))

            plt.plot(
                t_list.detach().cpu(),
                    [i.detach().cpu() for i in  vt_hat_A_error_list], label=f'A {sum(vt_hat_A_error_list)/len(vt_hat_A_error_list):.3f}')
            plt.plot(
                t_list.detach().cpu(), 
                    [i.detach().cpu() for i in  vt_hat_B_error_list], label=f'B {sum(vt_hat_B_error_list)/len(vt_hat_B_error_list):.3f}')
            plt.plot(
                t_list.detach().cpu(), 
                    [i.detach().cpu() for i in  vt_hat_diff_error_list], '--', label='diff')
            
            plt.legend()
            plt.grid()
            plt.title(f'vt_hat estimation MSE (D={D}) (t_kl_sampling_scheme={t_kl_sampling_scheme})')
            plt.tight_layout()
            plt.savefig(
                        Path(f'../{sfolder}/vhat_mse_upsample{upsample}_KL_ratio{KL_ratio:.3f}.pdf')
                        )
            plt.close()

            print('plot saved to ', Path(f'../{sfolder}/vhat_mse_upsample{upsample}_KL_ratio{KL_ratio:.3f}.pdf'))
            

        return KL_sample_mean.item(), KL_sample_std.item(), KL_est # samples_mean, samples_std



    

    @torch.no_grad()
    def kl_estimate_noisediffchannel(self, 
                    X1_data_A, 
                    
                    # ---- stats ----
                    upsample,
                    Phi,
                    lam_k_K,

                    analytic_stats=None, # mean_hat_A, mean_hat_B, lam_k_C = analytic_stats
                                        
                    # ---- Integration over FFM's t ----
                    n_t = 100,  
                    t_kl_sampling_scheme = False,  
                    # ---- (diff of) v_t 's source ----
                    # way 1
                    vdiff_source = 'sample',
                    v_source = 'neural_network',


                    sfolder = '',

                    X0_times = 1,
 

                    ):
        
       
        print(f"loop noise sampling for {X0_times} times")


        ############################################################
        print(f'to compute KL at upsample={upsample}')
        n_kl_samples = X1_data_A.shape[0]


        if v_source == 'neural_network':
            print("check boundary v_1(x) at t=1: =x?")
            Xt_data_A = X1_data_A
            t_inp     = torch.tensor([1.0],   device=device).expand(n_kl_samples)
 


            f_A = self.ffm_model_A.vt_from_output( self.ffm_model_A.prediction )
            f_B = self.ffm_model_B.vt_from_output( self.ffm_model_B.prediction )


            try:
                
                vt_data_A = f_A(t_inp, Xt_data_A.transpose(-1, -2), 
                                batch_y = ( torch.ones(n_kl_samples).to(Xt_data_A.device) * 0 ).long(),
                                model = self.ffm_model_A.model,
                                ).transpose(-1, -2)  
                vt_data_B = f_B(t_inp, Xt_data_A.transpose(-1, -2),
                                batch_y = ( torch.ones(n_kl_samples).to(Xt_data_A.device) * 1 ).long(),
                                ).transpose(-1, -2)  
                 
            except:

                breakpoint()

            
            plot_real_vs_fake(
                            y_real=X1_data_A, 
                            y_fake=vt_data_A, 
                            save_path=Path(f'../{sfolder}/v1Ax1A_vs_x1A.pdf')
                            )
            plot_real_vs_fake(
                            y_real=X1_data_A, 
                            y_fake=vt_data_B, 
                            save_path=Path(f'../{sfolder}/v1Bx1A_vs_x1A.pdf')
                            )
            
        kl_domain = 'freq'   # TODO
        print('kl_domain', kl_domain)
        
        kl_x0_t_log = []
        for X0_time in tqdm(range(X0_times), desc="X0", ncols=80):
            
 
            print(f"sample {n_kl_samples} XO per X0_time")

            X0_hat, X0_data = self.ffm_model_A.gp.sample(n_samples=n_kl_samples,   upsample=upsample,
                        return_hat = True
                        )  
            
     
            
            lam_k_K = self.ffm_model_A.gp.lam_k 
            lam_k_K = lam_k_K[None, :, :] 
            lam_k_K_sqrt = torch.sqrt( lam_k_K )
            # Phi = self.ffm_model_A.gp.Phi


            M = X1_data_A.shape[1] // upsample 
            # assert X1_data_A.shape[1] == Phi.shape[0]
            D = X1_data_A.shape[2] 
          

            vt_data_A_GT = X1_data_A - X0_data

            
          


            # ---- Integration over FFM's t ----
            t_list_returns = self.t_list_compute(n_t=n_t, t_kl_sampling_scheme=t_kl_sampling_scheme)
            if t_kl_sampling_scheme == 'uniform':
                t_list, sum_way = t_list_returns
            else:
                t_list, sum_way, normalizing_constant = t_list_returns

            t_list = t_list.detach().cpu().numpy().tolist() 
            print('t_list =', [round(float(x), 3) for x in t_list])
 

            ###########################################################################
            # ---- Integration over FFM's t ----
            # Loop over t only, thanks to the closed-form expectation wrt Xt

            kl_result = []
            
          
            for t in t_list: # tqdm(t_list, desc="t", ncols=80):
                 
                t_inp     = torch.tensor([
                    t
                ],   device=device).expand(n_kl_samples)

                

                Xt_data_A = t * X1_data_A + (1.0 - t) * X0_data

 





                try:
                    
                    vt_data_A = f_A(t_inp, Xt_data_A.transpose(-1, -2), 
                                    batch_y = ( torch.ones(n_kl_samples).to(Xt_data_A.device) * 0 ).long(),
                                    ).transpose(-1, -2)  
                    vt_data_B = f_B(t_inp, Xt_data_A.transpose(-1, -2),
                                    batch_y = ( torch.ones(n_kl_samples).to(Xt_data_A.device) * 1 ).long(),
                                    ).transpose(-1, -2)  

                    

                except:
                    breakpoint()


 


                
                # f =  self.ffm_model_A.vt_from_output( self.ffm_model_A.prediction  )
                # img = Xt_data_A.transpose(-1, -2)

                # vt_data_A = f(t_inp, img, self.ffm_model_A.model).transpose(-1, -2)
                # vt_data_B = f(t_inp, img, self.ffm_model_B.model).transpose(-1, -2)
               
            

                # if t_kl_sampling_scheme == "importance_sampling_t*(1-t)" and self.ffm_model_A.prediction == 'vt':
                #     # transform vt to mt
                #     vt_data_A = (Xt_data_A - vt_data_A) / (1-t)
                #     vt_data_B = (Xt_data_A - vt_data_B) / (1-t)
 

                if kl_domain == 'freq':
                    vt_hat_A = torch.fft.rfft(vt_data_A, dim=1, norm="ortho")  
                    vt_hat_B = torch.fft.rfft(vt_data_B, dim=1, norm="ortho")  

 

                    vdiffk_array = vt_hat_A - vt_hat_B

                    # without this, then identical to kl_domain = 'time'
                    vdiffk_array /= lam_k_K_sqrt  # TODO

                    

                    
                  
                    # because rfft only real half
                    v_diff_2_mode0     =   torch.abs(vdiffk_array[:, 0:1, :]) ** 2 
                    v_diff_2_modeNstar = ( torch.abs(vdiffk_array[:, 1:, :])  ** 2 ) * 2
                    v_diff_2           = torch.cat([ v_diff_2_mode0, v_diff_2_modeNstar ] , dim=1 )

               
                elif kl_domain == 'time':
                    vdiffk_array = vt_data_A - vt_data_B
                    v_diff_2   = torch.abs(vdiffk_array)**2

        
                if kl_domain == 'freq': 
                    kl_nocoeft = v_diff_2 
                    
                    kl_nocoeft = kl_nocoeft[:, :, : ]   # TODO #64


                elif kl_domain == 'time': 
                    kl_nocoeft = v_diff_2 

                     
                # torch.save(kl_nocoeft.mean(dim=0).detach().cpu()  , 
                #            f"NC_{int(t*100)}_tigon.pt"  ) 

                
                
                # check cvg over modes
                # kl_nocoeft.mean(dim=0).mean(dim=-1).cumsum(dim=0)

                # sum over freq 
                kl_nocoeft = torch.sum( 
                    kl_nocoeft, 
                    dim = -2 
                ) 
                assert kl_nocoeft.shape == (n_kl_samples, D) 
                

                if t_kl_sampling_scheme == 'uniform':
                    if 'vt' in self.ffm_model_A.prediction :
                        coef_t = t / (1. - t)  
                        # print(t.item(), coef_t.item())
                    elif self.ffm_model_A.prediction == 'mt':
                        coef_t = t * (1. - t)  

                    # coef_t = 1.0  
                else:
                    coef_t = normalizing_constant  # 0.99: 6.9078,

                # breakpoint()
                kl_coeft = coef_t * kl_nocoeft 
                kl_coeft = torch.real(kl_coeft) 

                kl_result.append( kl_coeft ) 

                
        
        
            kl_result = torch.stack(kl_result, dim=1) 
            assert kl_result.shape == (n_kl_samples, len(t_list), D)

            print_kl_list = []
            for t, kl_t in zip(t_list, kl_result.mean(dim=0).sum(dim=-1)):
                # print(f"t={t:.3f}: kl_t={kl_t.item():.6f}")
                print_kl_list.append( kl_t.item() )
            print('kl_list =', [round(float(x), 3) for x in print_kl_list])

            # sum  over D  
            kl_result = torch.sum(kl_result, dim=-1, keepdim=True)
            assert kl_result.shape == (n_kl_samples, len(t_list), 1)

            kl_x0_t_log.append(kl_result)

        
        kl_x0_t_log = torch.cat(kl_x0_t_log, dim=-1)
        assert kl_x0_t_log.shape == (n_kl_samples, len(t_list), X0_times)

        # mean over repeat X0_times = 1
        kl_result = torch.mean( kl_x0_t_log , dim=-1, keepdim=True )  # average over X0_time
        assert kl_result.shape == (n_kl_samples, len(t_list), 1)

        # Sum over t
        KL_est = integration_t(kl_result, t_list, sum_way=sum_way)
        assert KL_est.shape == (n_kl_samples, 1)
        KL_est = KL_est.squeeze(-1)
        assert KL_est.shape == (n_kl_samples,)

        
        


        # Mean over n_kl_samples
        KL_sample_std =  torch.std(KL_est) 
        KL_sample_mean = torch.mean(KL_est)


         
        # KL comparison
        print("\n=== Hyperparameters ===")
        hyper = dict(
            upsample=upsample, 
            n_t=n_t,
            t_kl_sampling_scheme=t_kl_sampling_scheme,
            
        )
        self.print_rows(hyper)

        print("\n=== KL result ===")
 
        result_kl = dict(
            KL_Est = KL_sample_mean.item(), 
        )

        self.print_rows(result_kl)

     
        return KL_sample_mean.item(), KL_sample_std.item(), KL_est # samples_mean, samples_std




    
    @torch.no_grad()
    def kl_estimate_gen(self, 
                    X1_data_A, 
                    
                    # ---- stats ----
                    upsample,
                    Phi,
                    lam_k_K,

                    analytic_stats=None, # mean_hat_A, mean_hat_B, lam_k_C = analytic_stats
                                        
                    # ---- Integration over FFM's t ----
                    n_t = 100,  
                    t_kl_sampling_scheme = False,  
                    # ---- (diff of) v_t 's source ----
                    # way 1
                    vdiff_source = 'sample',
                    v_source = 'neural_network',

                    sfolder = '',

                    ):
        
       

        torch.set_grad_enabled(False)
        ############################################################
        print(f'to compute KL at upsample={upsample}')

        n_kl_samples = X1_data_A.shape[0]

        X0_hat, X0_data = self.ffm_model_A.gp.sample(n_samples=n_kl_samples,   upsample=upsample,
                    return_hat = True
                    ) # [0]
        

        
        
        lam_k_K = self.ffm_model_A.gp.lam_k
        Phi = self.ffm_model_A.gp.Phi


        
        M = X1_data_A.shape[1] // upsample
        assert X1_data_A.shape[1] == Phi.shape[0]
        D = X1_data_A.shape[2]
        N = (Phi.shape[1] - 1) // 2
        assert N < int(M*upsample)/2, "Nyquist condition is not satisfied!"  
 

        X1_hat_A = project_to_basis(X1_data_A, Phi)
        
        # assert abs(self.ffm_model_A.gp.Phi - Phi).max() == 0

        assert X1_hat_A.shape == X0_hat.shape
 
        ############################################################
        num_steps = 100

        timesteps = torch.linspace(
            # 1, 0, 
            # 0.96, 0, 
            0.96, 0, 
            num_steps) 
        timesteps = timesteps.tolist() 
        timesteps = timesteps[::-1] 

        t_list = timesteps[:-1]
        sum_way = 'riemann'
        img = X0_data


        denoise_strategy = 'denoise_midpoint' # 'denoise' # 'denoise_midpoint'
        ###########################################################################
        # ---- Integration over FFM's t ----
 
        kl_result = []
        with torch.no_grad():
            img = img.transpose(-1, -2)
            for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                if denoise_strategy == 'denoise':
                    t_vec = torch.full((img.shape[0],), 
                                    t_curr, 
                                    dtype=img.dtype, device=img.device)
                    
                    # vt_data_A = self.ffm_model_A.model(t_vec, img)
                    # vt_data_B = self.ffm_model_B.model(t_vec, img)

                    f =  self.ffm_model_A.vt_from_output( self.ffm_model_A.prediction  )
                    vt_data_A = f(t_vec, img, self.ffm_model_A.model)
                    vt_data_B = f(t_vec, img, self.ffm_model_B.model)

                    img = img + (t_prev - t_curr)   * vt_data_A
                elif denoise_strategy == 'denoise_midpoint':
                    t_vec = torch.full((img.shape[0],), 
                                    t_curr, 
                                    dtype=img.dtype, device=img.device)
                    
                    f =  self.ffm_model_A.vt_from_output( self.ffm_model_A.prediction  )
                    vt_data_A = f(t_vec, img, self.ffm_model_A.model)
                    vt_data_B = f(t_vec, img, self.ffm_model_B.model)

                    img_mid = img + (t_prev - t_curr) / 2 * vt_data_A

                    t_vec_mid = torch.full((img.shape[0],), 
                                        t_curr + (t_prev - t_curr) / 2, 
                                        dtype=img.dtype, device=img.device) 
            
                    pred_mid = f(t_vec_mid, img_mid, self.ffm_model_A.model) 
                
                    img = img + (t_prev - t_curr) * pred_mid


                

                t = t_curr
                
           
                vt_hat_A = project_to_basis(vt_data_A.transpose(-1, -2), Phi)
                vt_hat_B = project_to_basis(vt_data_B.transpose(-1, -2), Phi)
                vdiffk_array = vt_hat_A - vt_hat_B
                v_diff_2   = torch.abs(vdiffk_array)**2
                assert v_diff_2.shape == (n_kl_samples, 2*N+1, D) 
                kl_nocoeft = torch.sum( 
                    v_diff_2 / lam_k_K[None, :, None] , 
                    dim = -2 
                )
                assert kl_nocoeft.shape == (n_kl_samples, D)
                if t_kl_sampling_scheme == 'uniform':
                    if 'vt' in self.ffm_model_A.prediction :
                        coef_t = t / (1. - t)  
                    elif self.ffm_model_A.prediction == 'mt':
                        coef_t = t * (1. - t)  
                else:
                    raise ValueError(f"Got unknown t_kl_sampling_scheme: {t_kl_sampling_scheme}")
                kl_coeft = coef_t * kl_nocoeft 
                kl_coeft = torch.real(kl_coeft)
                kl_result.append( kl_coeft )
        
        
        kl_result = torch.stack(kl_result, dim=1) 

        assert kl_result.shape == (n_kl_samples, len(t_list), D)

        print_kl_list = []
        for t, kl_t in zip(t_list, kl_result.mean(dim=0).mean(dim=-1)):
            print(f"t={t:.3f}: kl_t={kl_t.item():.6f}")
            print_kl_list.append(kl_t.item())
        print('t', t_list)
        print('print_kl_list', print_kl_list)
        
 
        # Sum over t
        KL_est = integration_t(kl_result, t_list, sum_way=sum_way)
        integration_t(kl_result, t_list, sum_way=sum_way,
                        device=device)
        assert KL_est.shape == (n_kl_samples, D)


        # Sum  over D 
        KL_est = torch.sum(KL_est, dim=-1)
        assert KL_est.shape == (n_kl_samples,)


        # Mean over samples
        KL_sample_std =  torch.std(KL_est) 
        KL_sample_mean = torch.mean(KL_est)


        
 

        # KL comparison
        print("\n=== Hyperparameters ===")
        hyper = dict(
            upsample=upsample, 
            n_t=n_t,
            t_kl_sampling_scheme=t_kl_sampling_scheme,
            
        )
        self.print_rows(hyper)

        print("\n=== KL result ===")
        result_kl = dict(
            KL_Est = KL_sample_mean.item(), 
        )
        self.print_rows(result_kl)
 

        return KL_sample_mean.item(), KL_sample_std.item(), KL_est # samples_mean, samples_std


  
    @torch.no_grad()
    def kl_estimate_gen_at(self, 
                    X1_data_A, 
                    
                    # ---- stats ----
                    upsample,
                    Phi,
                    lam_k_K,

                    analytic_stats=None, # mean_hat_A, mean_hat_B, lam_k_C = analytic_stats
                                        
                    # ---- Integration over FFM's t ----
                    n_t = 100,  
                    t_kl_sampling_scheme = False,  
                    # ---- (diff of) v_t 's source ----
                    # way 1
                    vdiff_source = 'sample',
                    v_source = 'neural_network',

                    # # way 2
                    # vdiff_source = 'sample'
                    # v_source     = 'analytic'  

                    # # way 3
                    # vdiff_source = 'analytic'  , 

                    sfolder = '',

                    ):
        
       

        torch.set_grad_enabled(False)
        ############################################################
        print(f'to compute KL at upsample={upsample}')

        n_kl_samples = X1_data_A.shape[0]

        X0_hat, X0_data = self.ffm_model_A.gp.sample(n_samples=n_kl_samples,   upsample=upsample,
                    return_hat = True
                    ) # [0]
        

   
        
        lam_k_K = self.ffm_model_A.gp.lam_k
        Phi = self.ffm_model_A.gp.Phi


        
        M = X1_data_A.shape[1] // upsample
        assert X1_data_A.shape[1] == Phi.shape[0]
        D = X1_data_A.shape[2]
        N = (Phi.shape[1] - 1) // 2
        assert N < int(M*upsample)/2, "Nyquist condition is not satisfied!"  


        X1_hat_A = project_to_basis(X1_data_A, Phi)
        
        # assert abs(self.ffm_model_A.gp.Phi - Phi).max() == 0

        assert X1_hat_A.shape == X0_hat.shape
 
        ############################################################
        num_steps = 100

        timesteps = torch.linspace(
            # 1, 0, 
            1, 0, 
            num_steps) 
        timesteps = timesteps.tolist() 
        timesteps = timesteps[::-1] 

        t_list = timesteps[:-1]
        sum_way = 'riemann'
        img = X0_data


        denoise_strategy = 'denoise_midpoint' # 'denoise' # 'denoise_midpoint'
        ###########################################################################
        # ---- Integration over FFM's t ----
 
        kl_result = []
        with torch.no_grad():
            img = img.transpose(-1, -2)
            for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
               
                if denoise_strategy == 'denoise_midpoint':
                    t_vec = torch.full((img.shape[0],), 
                                    t_curr, 
                                    dtype=img.dtype, device=img.device)

                    f =  self.ffm_model_A.vt_from_output( self.ffm_model_A.prediction  )
                    pred_A = f(t_vec, img, self.ffm_model_A.model)
                    pred_B = f(t_vec, img, self.ffm_model_B.model)
                    # pred_A = self.ffm_model_A.model(t_vec, img)
                    # pred_B = self.ffm_model_B.model(t_vec, img)
               

                    img_mid_A = img + (t_prev - t_curr) / 2 * pred_A
                    img_mid_B = img + (t_prev - t_curr) / 2 * pred_B

                    t_vec_mid = torch.full((img.shape[0],), 
                                        t_curr + (t_prev - t_curr) / 2, 
                                        dtype=img.dtype, device=img.device)

                  
                    vt_data_A = f(t_vec_mid, img_mid_A, self.ffm_model_A.model)
                    vt_data_B = f(t_vec_mid, img_mid_B, self.ffm_model_B.model)
                    # vt_data_A = self.ffm_model_A.model(t_vec_mid, img_mid_A)
                    # vt_data_B = self.ffm_model_B.model(t_vec_mid, img_mid_B)
                
                    img = img + (t_prev - t_curr) * vt_data_A

                    at_data_A = (vt_data_A - pred_A) / ( (t_prev - t_curr) / 2 )
                    at_data_B = (vt_data_B - pred_B) / ( (t_prev - t_curr) / 2 )

 
                t = t_curr
                
           
                at_hat_A = project_to_basis(at_data_A.transpose(-1, -2), Phi)
                at_hat_B = project_to_basis(at_data_B.transpose(-1, -2), Phi)
                adiffk_array = at_hat_A - at_hat_B
                a_diff_2   = torch.abs(adiffk_array)**2
                assert a_diff_2.shape == (n_kl_samples, 2*N+1, D) 
                kl_nocoeft = torch.sum( 
                    a_diff_2 / lam_k_K[None, :, None] , 
                    dim = -2 
                )
                assert kl_nocoeft.shape == (n_kl_samples, D)
                if t_kl_sampling_scheme == 'uniform':

                    assert 'vt' in self.ffm_model_A.prediction  
                    coef_t = t * (1. - t)  
                    
                else:
                    raise ValueError(f"Got unknown t_kl_sampling_scheme: {t_kl_sampling_scheme}")
                kl_coeft = coef_t * kl_nocoeft 
                kl_coeft = torch.real(kl_coeft)
                kl_result.append( kl_coeft )
        
        
        kl_result = torch.stack(kl_result, dim=1) 

        assert kl_result.shape == (n_kl_samples, len(t_list), D)

        for t, kl_t in zip(t_list, kl_result.mean(dim=0).flatten()):
            print(f"t={t:.3f}: kl_t={kl_t.item():.6f}")
 
        # Sum over t
        KL_est = integration_t(kl_result, t_list, sum_way=sum_way)
        integration_t(kl_result, t_list, sum_way=sum_way,
                        device=device)
        assert KL_est.shape == (n_kl_samples, D)


        # Sum  over D 
        KL_est = torch.sum(KL_est, dim=-1)
        assert KL_est.shape == (n_kl_samples,)


        # Mean over samples
        KL_sample_std =  torch.std(KL_est) 
        KL_sample_mean = torch.mean(KL_est)


        
 

        # KL comparison
        print("\n=== Hyperparameters ===")
        hyper = dict(
            upsample=upsample, 
            n_t=n_t,
            t_kl_sampling_scheme=t_kl_sampling_scheme,
            
        )
        self.print_rows(hyper)

        print("\n=== KL result ===")
        result_kl = dict(
            KL_Est = KL_sample_mean.item(), 
        )
        self.print_rows(result_kl)
 

        return KL_sample_mean.item(), KL_sample_std.item(), KL_est  

