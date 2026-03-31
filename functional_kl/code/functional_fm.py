import numpy as np
import torch
from torchdiffeq import odeint
from util.util import reshape_for_batchwise, plot_loss_curve, plot_real_vs_fake, plot_real_vs_fake_2d, plot_spectrum_comparison
import time
from util.util import save_checkpoint
import os
import wandb
from torchcfm.optimal_transport import OTPlanSampler
import copy
import matplotlib.pyplot as plt
from kl_functions_torch import sample_trunc_t_over_1mt, project_to_basis # *
# basis = 'fourier'
from neuralop.layers.padding import DomainPadding
from functools import partial

class FFMModel:
    def __init__(self, 
                model, 
                sample_X0_data_func, 
                D, 
                x_dim,

                device='cuda', 

                prediction='v',
                t_train_sampling_scheme="importance_sampling_t/(1-t)",

                loss_lam_time = 1.0,
                loss_decay = False,

                curriculum_sampling=False,
                curriculum_switch_frac=0.4,
                curriculum_logit_mean=0.8,
                curriculum_logit_std=1.0,

                ):
        self.model = model
        self.device = device
        self.gp = sample_X0_data_func

        self.n_channels = D
        self.x_dim=x_dim

        self.prediction = prediction
        self.t_train_sampling_scheme = t_train_sampling_scheme


        self.loss_lam_time = loss_lam_time
        self.loss_decay = loss_decay

        self.curriculum_sampling = curriculum_sampling
        self.curriculum_switch_frac = curriculum_switch_frac
        self.curriculum_logit_mean = curriculum_logit_mean
        self.curriculum_logit_std = curriculum_logit_std
  
 
    
    def simulate(self, t, x_data):
        # t: [batch_size,]
        # x_data: [batch_size, n_channels, *dims]
        # samples from p_t(x | x_data)
        
        
        # Sample GP noise  from prior GP
        batch_size = x_data.shape[0]

        noise = self.gp.sample(n_samples=batch_size).to(dtype = x_data.dtype, device = x_data.device)
        
        
       
        self.ot_method = "independent" # "independent" # "exact_ot_cfm"
        self.sigma_min = 1e-4 # sigma = [0.0, 5e-4, 0.5, 1.5, 0, 1][0]

        if self.ot_method == "sb_cfm": 
            noise, x_data = OTPlanSampler(method="sinkhorn", reg=2 * (self.sigma_min**2)).sample_plan(noise, x_data)
        elif self.ot_method == "exact_ot_cfm":
            noise, x_data = OTPlanSampler(method="exact").sample_plan(noise, x_data)
    

        # Construct mean/variance parameters
        t = reshape_for_batchwise(t, 1 + self.x_dim)

         
        
        xt = t * x_data + (1.-t) * noise
        vt = x_data - noise

 
        
        if self.ot_method == "exact_ot_cfm":
   
            epsilon = self.gp.sample(n_samples=batch_size).to(dtype = x_data.dtype, device = x_data.device) 
       
            sigma_t = self.sigma_min 
            xt = xt + sigma_t * epsilon


        elif self.ot_method == "sb_cfm":
            epsilon = self.gp.sample(n_samples=batch_size).to(dtype = x_data.dtype, device = x_data.device) 

            sigma_t = self.sigma_min * torch.sqrt(t * (1 - t))
            xt = xt + sigma_t * epsilon

            vt = (
                (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
                * (sigma_t * epsilon)  # (xt - (t * x_data + (1. - t) * noise))
                + vt    # + x1 - x0
            )
    

        return xt, vt
    

    def logit_normal_dist(self, batch_size, device, 
                          P_mean=0.0, P_std=1.0):
        t = torch.normal(
                mean=P_mean, std=P_std,  # mean=0.8, std=0.8, 
                size=(batch_size,), device=device, 
                generator=None)
        t = torch.nn.functional.sigmoid(t)
        return t


    
    def batch_loss( self, 
                   model, 
                   batch_tuple,  ):
        
        device = self.device
    
        # batch, X1_mean_hat = batch_tuple


        batch = batch_tuple[0]
        batch_y = batch_tuple[1]

        batch = batch.to(device)
        batch_y = batch_y.to(device)

        # Sample a random timestep for each image
        batch_size = batch.shape[0]
        normalizing_constant = None

        # self.t_train_sampling_scheme = "logit_normal"
        self.tr_uniform = False

        # Curriculum sampling: override time sampling based on epoch progress
        if self.curriculum_sampling:
            frac = self.ep_cur / self.ep_total
            if frac <= self.curriculum_switch_frac:
                # Phase 1: logit-normal (structure learning)
                t = self.logit_normal_dist(batch_size, device,
                                           P_mean=self.curriculum_logit_mean,
                                           P_std=self.curriculum_logit_std)
            else:
                # Phase 2: uniform (boundary refinement)
                t = torch.rand(batch_size, device=device)

        elif  self.t_train_sampling_scheme == "logit_normal": # for weighting schemes where we sample timesteps non-uniformly
            t = self.logit_normal_dist(batch_size, device, 
                          P_mean=0.0, P_std=1.0)
            
            if self.tr_uniform:
                # 10% random tr samples
                gen = torch.Generator(device=t.device)
                t_unif = torch.rand((batch_size,), device=device, generator=gen)
                unif_mask = t_unif < 0.1
                t = torch.where(
                    unif_mask,
                    t_unif,
                    t,
                )
           

        elif  self.t_train_sampling_scheme == "uniform":
            # t ~ Unif[0, 1)
            t = torch.rand(batch_size, device=device)

        elif self.t_train_sampling_scheme == "importance_sampling_t/(1-t)":
            # ---- Integration over FFM's t ----
            t_eps = 0.995 # 0.996
            t, normalizing_constant  = sample_trunc_t_over_1mt(t_eps, batch_size, return_Z=True,)

        elif self.t_train_sampling_scheme == "importance_sampling_t*(1-t)":
            t = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(device)
            normalizing_constant = 1.0 / 6.0

            # t = t.clamp(0.0, 1.0 - 1e-3)
        
        else:
            raise ValueError(f"Unknown  t_train_sampling_scheme: { self.t_train_sampling_scheme}")



        # Simluate p_t(x | x_1)
        with torch.no_grad():
            x_noisy, v_t = self.simulate(t, batch)
            x_noisy = x_noisy.to(device)
            v_t  = v_t.to(device)   

        if 'vt' in self.prediction:
            target = v_t
        elif self.prediction == 'mt':
            target = (x_noisy - v_t) / (1.0-t)
        elif  self.prediction == 'x1':
            target = batch
        




        else:
            raise ValueError(f"Unknown  prediction: { self.prediction}")


        # Get model output
        if x_noisy.ndim == 3:

            f = self.vt_from_output( self.prediction )
            model_out = f( t, x_noisy.transpose(-1, -2) ,
            batch_y=batch_y,
                          ).transpose(-1, -2)
           

 
        elif x_noisy.ndim == 4:
            model_out = model(t, x_noisy.permute(0, 3, 1, 2).contiguous() ).permute(0, 2, 3, 1).contiguous()      

        
        # Loss
        lam_time = self.loss_lam_time
        
        if self.loss_decay:
            # linear decay: lam_time goes from loss_lam_time → 0.2 over training
            decay = 1.0 - 0.8 * self.ep_cur / self.ep_total
            lam_time *= decay

        loss, loss_time, loss_freq = self.loss_fn(pred=model_out, gt=target,
                            lam_time = lam_time, lam_freq = 1.0 - lam_time,
                            # lam_time = 0.0, lam_freq = 1.0
                            )

        if normalizing_constant is not None: # "importance_sampling" in self.t_train_sampling_scheme:
            loss = loss * normalizing_constant

        return loss, loss_time, loss_freq
    

    def loss_weight_formulation(self, diff):
        # spectrum weight matrix
        # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
        # matrix_tmp = (recon_freq - real_freq) ** 2
        # matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) 
        matrix_tmp = torch.sqrt(diff) 
        matrix_tmp = matrix_tmp / matrix_tmp.max(1, keepdim=True)  

        matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
        matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
        weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        return weight_matrix
 

    def loss_fn(self, pred, gt, lam_time=1.0, lam_freq=0.0):
        
        loss_time = 0
        if lam_time > 0.0:
            loss_time = (pred-gt).pow(2).mean()
       


 
        loss_freq = 0.0
        if lam_freq > 0.0:
      

            pred_hat = torch.fft.rfft(pred, dim=1, norm="ortho")      # (B, N//2+1)   project_to_basis(pred,   self.gp.Phi)
            gt_hat   = torch.fft.rfft(gt,   dim=1, norm="ortho")  

             
            pred_hat = torch.view_as_real(pred_hat)  # [16, 376, 3, 2]
            gt_hat = torch.view_as_real(gt_hat)  # [16, 376, 3, 2]
            # frequency distance using (squared) Euclidean distance
            tmp = (pred_hat - gt_hat) ** 2
            diff = tmp[..., 0] + tmp[..., 1] # [16, 376, 3])


            # your 1D tensor (indices sorted descending)
            diff_record =  diff.mean(dim=0).mean(dim=-1).detach() 

            loss_freq = diff 
            loss_freq = diff.mean()

            diff /= self.gp.lam_k[None, :, :]

  
    
        total = lam_time * loss_time + lam_freq * loss_freq
        return total, loss_time, loss_freq
 

    def train(self, train_loader, optimizer, epochs, 
                scheduler=None, test_loader=None, eval_int=0, 
           
                generate=False, save_path=None, 
            
                ):

        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)

        model = self.model
        tolerance = 0


        self.ep_total = epochs
        for ep in range(1, epochs+1):
            self.ep_cur = ep
            ##### TRAINING LOOP
            model.train()

            t0 = time.time()
            tr_loss = 0.0
            tr_loss_time = 0.0
            tr_loss_freq = 0.0

            for batch_idx, batch_tuple in enumerate(train_loader, start=1):

                loss, loss_time, loss_freq = self.batch_loss(model,  batch_tuple,  )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.optimizer.zero_grad()

                tr_loss += loss.item()
                tr_loss_time += loss_time.item() if torch.is_tensor(loss_time) else loss_time
                tr_loss_freq += loss_freq.item() if torch.is_tensor(loss_freq) else loss_freq
            n_batches = len(train_loader)
            tr_loss /= n_batches
            tr_loss_time /= n_batches
            tr_loss_freq /= n_batches
            tr_losses.append(tr_loss)

            if scheduler: scheduler.step()

            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')


            # ---- wandb: log train metrics ----
            log_dict = {
                "epoch": ep,
                "train/loss": tr_loss,
                "train/loss_time": tr_loss_time,
                "train/loss_freq": tr_loss_freq,
            }


            # robust lr fetch
            try:
                lr = optimizer.param_groups[0]["lr"]
            except Exception:
                lr = None
            if lr is not None:
                log_dict["lr"] = lr


            ##### EVAL LOOP
            if ((eval_int > 0 and (ep % eval_int == 0)) or (ep == epochs)):
                eval_eps.append(ep)

                with torch.no_grad():
                    model.eval()

                    # switch to EMA parameters
                    if hasattr(optimizer, 'swap_parameters_with_ema'):
                        optimizer.swap_parameters_with_ema(store_params_in_ema=True)


                    if evaluate:
                        t0 = time.time()
                        te_loss = 0.0
                
                        for batch_tuple in test_loader:
                            loss, _, _ = self.batch_loss(model,  batch_tuple,  )

                            te_loss += loss.item()
                        te_loss /= len(test_loader)

                        te_losses.append(te_loss)

                        t1 = time.time()
                        epoch_time = t1 - t0
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {epoch_time:.2f} (s)')
 

                        # ---- wandb: log val metrics ----
                        log_dict["val/loss"] = te_loss
                      
                        save_ckpt = True

                            


 
                    else:
                        save_ckpt = True


                    # genereate samples during training?
                    if generate:

                        batch_xA = train_loader.dataset[:generate][0]
                        batch_xB = train_loader.dataset[-generate:][0]
              

 
                       
                        samples, x0 = self.sample(  
                                    n_samples= generate, # 5_000 if  upsample != 16 else 1_000 , #
                                    upsample = 1,
                                    return_x0=True) # [n_gen_samples, n_channels, n_x * upsample]

                        
                        # if samples.ndim == 3:

                        

                        plot_real_vs_fake_eval_A = plot_real_vs_fake(
                                    y_real=batch_xA, 
                                    y_fake=samples[0], 
                                    save_path=None,  
                                    )
                        plot_real_vs_fake_eval_B = plot_real_vs_fake(
                                    y_real=batch_xB, 
                                    y_fake=samples[1], 
                                    save_path=None,  
                                    )
                        log_dict['gen_samples_eval_A'] = plot_real_vs_fake_eval_A
                        log_dict['gen_samples_eval_B'] = plot_real_vs_fake_eval_B
                        
                        


                        # print(x0.shape, samples.shape, batch.shape)
                        plot_spectrum_comparison_eval_A  = plot_spectrum_comparison(x1_time=batch_xA, x1_time_gen=samples[0], x0_time=x0, 
                                    save_path=None, 
                                    )
                        plot_spectrum_comparison_eval_B  = plot_spectrum_comparison(x1_time=batch_xB, x1_time_gen=samples[1], x0_time=x0, 
                                    save_path=None, 
                                    )
                        log_dict['spectrum_comparison_eval_A'] = plot_spectrum_comparison_eval_A
                        log_dict['spectrum_comparison_eval_B'] = plot_spectrum_comparison_eval_B

                            
                        # elif samples.ndim == 4:
                        #     plot_real_vs_fake_2d(
                        #             y_real=batch[0, :, :, 0].detach().cpu().numpy(), 
                        #             y_fake=samples[0, :, :, 0].detach().cpu().numpy(), 
                        #             save_path=save_path / 'imgs' /  f'gen_samples_epoch{ep}.pdf'
                        #             )
                            

                    if save_ckpt:
                        checkpoint_file = save_path / f'ckpt/epoch_{ep}.pt'
                        # torch.save(model.state_dict(), checkpoint_file)
                        save_checkpoint(ep, model, optimizer, scheduler, checkpoint_file=checkpoint_file)
                        print(f'Saved checkpoint at epoch {ep:d} to {checkpoint_file}')


 
                    # switch back to original parameters
                    if hasattr(optimizer, 'swap_parameters_with_ema'):
                        optimizer.swap_parameters_with_ema(store_params_in_ema=True)


                    if tolerance >= 20:
                        print(f'Early stopping at epoch {ep} due to no improvement in val loss for {tolerance} evals.')
                        return


  
            wandb.log(log_dict, step=ep)

            #### PLOT LOSS CURVE
            if evaluate:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            else:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf')

    
    

    def vt_from_output(self,  prediction):
        """
        Returns f(t, x) = dx/dt for odeint.
        `extra` can hold whatever you need to convert model output -> drift
        (e.g., schedule params, conditioning, etc.)
        """

        def f(
        t, x, 
        batch_y = None,
        model = self.model, 
        ):

            assert batch_y is not None

            # print(t)

       

            # prediction = "vt" # _from_substraction"
 

            if  prediction == "vt":

                if model.model_name == 'fno':
                    model_out = model(t, x)  
                elif model.model_name == 'transformer':
                    try:
                        model_out = model(x.to(t.dtype), t,
                        batch_y
                        )
                    except:
                        breakpoint()
                elif model.model_name == 'mino_t':
                    model_out = model(x.to(t.dtype), t, batch_y)


                # If your model already outputs drift (velocity), use directly:
                dxdt = model_out
  

            elif  prediction == "mt":
                # If model outputs some "m(t)" that is NOT dx/dt, convert it here.
                # You must define this mapping for your SDE/ODE parameterization.
                dxdt = x + (t - 1.0) * model_out

            elif  prediction == "x1":
                # If model predicts terminal/clean state x1, convert to drift here.
                dxdt = (model_out - x) / (1.0 - t)
           
            elif prediction == 'vt_from_substraction':
                # m_t = model_out
 

                if model.model_name == 'fno':
                    m_t = model(t, x)  
                    m_1 = model(torch.ones_like(t), x)     # mθ(x,1)  (same x, time=1)
                elif model.model_name == 'transformer':
                    try:
                        m_t = model(x.to(t.dtype), t,                  batch_y)
                        m_1 = model(x.to(t.dtype), torch.ones_like(t), batch_y)
                    except:
                        breakpoint()
                elif model.model_name == 'mino_t':
                    m_t = model(x.to(t.dtype), t, batch_y)
                    m_1 = model(x.to(t.dtype), torch.ones_like(t), batch_y)

                # while t.ndim < x.ndim:
                #     t = t.unsqueeze(-1)
            
                dxdt = x + ( m_t - m_1 ) 

           
                # with torch.no_grad():
                #     print("time emb diff:", (
                #         model.h_embedder(t) - model.h_embedder(torch.ones_like(t))
                #     ).abs().mean().item())
                    
               


                # dxdt = x + t * ( m_t - m_1 ) 
                # dxdt = torch.tanh(x) + t * ( m_t - m_1 ) # Jan8
                # dxdt = torch.tanh( ( t - 0.5 ) * 5 ) * x + (t + 0.05) * ( m_t - m_1 ) # Jan9

            elif prediction == 'vt_from_2boundary' :
                m_t = model_out

                while t.ndim < x.ndim:
                    t = t.unsqueeze(-1)

                g_t = torch.cos(0.5 * torch.pi * t)
                f_t = torch.sin(0.5 * torch.pi * t)
                h_t = torch.sin(torch.pi * t)

                C = torch.zeros_like(x) 
                dxdt = g_t * (C - x) + f_t * x + h_t * m_t

            else:
                raise ValueError(f"Unknown  prediction: { prediction}")


            # breakpoint()
            return dxdt

        return f

    
    def denoise(self, timesteps, img, f):

        img = img.transpose(-1, -2)
 
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), 
                               t_curr, 
                               dtype=img.dtype, device=img.device)
            
            pred = f(t_vec, img)

            img = img + (t_prev - t_curr)   * pred
             
        img = img.transpose(-1, -2)
        
        return img

    
    def denoise_midpoint(self, timesteps, img, f):

        img = img.transpose(-1, -2)

       
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), 
                               t_curr, 
                               dtype=img.dtype, device=img.device)
            
            pred = f(t_vec, img)

            img_mid = img + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((img.shape[0],), 
                                   t_curr + (t_prev - t_curr) / 2, 
                                   dtype=img.dtype, device=img.device)
     
            pred_mid = f(t_vec_mid, img_mid)
           
            img = img + (t_prev - t_curr) * pred_mid

         

        img = img.transpose(-1, -2)
        
        
        return img

        

    @torch.no_grad()
    def sample(self, 
               n_samples=1, 
               upsample=1, 
               n_eval=2, 
               return_path=False, rtol=1e-5, atol=1e-5, 
               return_x0 = False,
               ):
        # n_eval: how many timesteps in [0, 1] to evaluate. Should be >= 2. 

        
        
        method =  'euler' # 'dopri5'    

        denoise_strategies = {
            'denoise' : self.denoise,
          
            'denoise_midpoint' : self.denoise_midpoint,
        }

        # if method == 'dopri5':
           
        out_dict = {}
        for id in [1,0]:

            x0 = self.gp.sample(n_samples=n_samples, upsample=upsample).to(device = self.device)  
            # x0 = torch.randn_like(x0)  

            if method == 'dopri5':
                n_eval = 2
                t = torch.linspace(0, 1, n_eval, device=self.device)
            elif method == 'euler':
                n_eval = 100
                t = torch.linspace(0, 1, n_eval, device=self.device)

            

            f_noy = self.vt_from_output( self.prediction )
            f = partial(
                f_noy,
                batch_y = ( 
                    # torch.cat([
                    #     torch.ones(n_samples//2).to(x0.device),
                    #     torch.zeros(n_samples//2).to(x0.device)
                    # ])
                    torch.ones(n_samples).to(x0.device) * id
                      ).long(),
                model   = self.model,
            )

            out_path_id = odeint(func=f, 
                        y0=x0.transpose(-1, -2), 
                        t=t, 
                        method=method, rtol=rtol, atol=atol).transpose(-1, -2) 
            out_dict[id] = out_path_id[-1] 
        


        # if return_path:
        #     return out
        # else:

        if return_x0:
            return out_dict, x0
        return out_dict
        
        
        # elif  method in denoise_strategies.keys():
        #     if x0.ndim == 3:
            
        #         denoise_strategy = denoise_strategies[method]

        #         num_steps = 100

        #         timesteps = torch.linspace(1, 0, num_steps + 1)
        #         timesteps = timesteps.tolist()
        #         timesteps = timesteps[::-1]

        #         out = denoise_strategy(timesteps, x0, f)
        #         return out

        
