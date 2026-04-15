import torch
import torch.nn as nn
import math
import numpy as np
from diffusers import DDIMScheduler
from torch.distributions import Normal

# Initialize the DDIM scheduler with default parameters


class DDPM(nn.Module):
    def __init__(self, 
                 beta_min=0.1,
                 beta_max=20,
                 T=1000,
                 device = "cuda"):
        super(DDPM, self).__init__() 

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T  # Total number of discrete timesteps
        self.device = device
        # Precompute betas, alphas, and alpha_bars for discrete timesteps
        self.betas = torch.linspace(1e-4, 0.02, T).to(self.device)
        if self.T!=1000:
            self.betas = torch.linspace(1e-4, 0.2, T).to(self.device)
            # betas = torch.linspace(-6, 6, T)
            # self.betas = torch.sigmoid(betas) * (1e-5 - 1e-2) + 1e-5
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.register_buffer('betas_buffer', self.betas)
        self.register_buffer('alphas_buffer', self.alphas)
        self.register_buffer('alpha_bars_buffer', self.alpha_bars)
        self.final_alpha_cumprod = self.alpha_bars[0] 

        


    def set_device(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)

    def beta_t(self, t):
        """
        Returns beta_t for a given discrete timestep torch.
        """
        return self.betas_buffer[t]

    def alpha_t(self, t):
        """
        Returns \alpha_t for a given discrete timestep torch.
        """
        return self.alphas_buffer[t]

    def alpha_bar_t(self, t):
        """
        Returns \bar{\alpha}_t for a given discrete timestep torch.
        """
        return self.alpha_bars_buffer[t]

    def marg_prob(self, t, x):
        """
        Returns the mean and standard deviation of the marginal probability P(x_t | x_0).
        """
        alpha_bar_t = self.alpha_bar_t(t)
        mean = torch.sqrt(alpha_bar_t)
        std = torch.sqrt(1 - alpha_bar_t)
        mean = mean.view(mean.shape[0], *([1] * (x.ndim - 1)))
        std = std.view(std.shape[0], *([1] * (x.ndim - 1)))
        return mean, std
    
    def q_sample(self, x_0, t):
        """
        Forward diffusion process: Sample x_t ~ q(x_t | x_0).
        """
        mean,std = self.marg_prob(t,x_0)
        z = torch.randn_like(x_0.float(), device=self.device)
        x_t = mean * x_0 + std * z
        return x_t, z, mean, std


    def q_posterior(self, x_0, x_t, t):
        """
        Compute the posterior q(x_{t-1} | x_t, x_0) for a given timestep torch.
        """
        if t == 0:
            raise ValueError("q_posterior is undefined for t=0.")

        alpha_bar_t = self.alpha_bar_t(t)
        alpha_bar_t_minus_1 = self.alpha_bar_t(t - 1)
        beta_t = self.beta_t(t)

        mean = (
            torch.sqrt(alpha_bar_t_minus_1) * x_0 + 
            (1 - alpha_bar_t_minus_1) * x_t / torch.sqrt(1 - alpha_bar_t)
        ) / torch.sqrt(alpha_bar_t_minus_1 + beta_t)

        std = torch.sqrt(beta_t * (1 - alpha_bar_t_minus_1) / (alpha_bar_t_minus_1 + beta_t))
        return mean, std

    


    


  


    def sample_uniform(self, shape, eps=0.0,max_eps =None):
        eps = int(self.T * eps )
        max_eps = eps if max_eps==None else int(max_eps * self.T)
        return torch.randint(low=eps, high=int(self.T-max_eps), size=shape, device=self.device)

    
    def sample_from_model(self,score_function, x0,c,guidance,num_steps, clip_gen=False,randomized=True,compute_kl=False,ema=False, eta=1.0,intervals=[-1,1]):
        """
        Generate a sample using DDIM.

        Parameters:
        - x_T: Initial noisy sample at timestep torch.
        - num_steps: Number of steps for the sampling schedule.
        - eta: Controls stochasticity; eta=0 is deterministic.

        Returns:
        - x_0: Generated sample at timestep 0.
        """
        
        # Create a time schedule from T to 0 with `num_steps` intervals
        sch = DDIMScheduler(num_train_timesteps=self.T, steps_offset=1)
    
        sch.set_timesteps(num_inference_steps=num_steps)

        timesteps = sch.timesteps

        x_t = x0
        ones = torch.ones((x0.size(0),),device=x0.device)
     
        kl=0
        for i in range(num_steps):
            
            t = timesteps[i]
            t_next = timesteps[i + 1] if i < len(timesteps)-1 else -1
  
            t_vec = (ones * t).long()
            t_next_vec = (ones*t_next).long()
            
            if guidance!=1.0 and guidance!=0.0:
     
                noise_pred, noise_pred_uncond = score_function(x_t,cond = c,t=t_vec,ema =ema ),score_function(x_t,cond = None,t=t_vec,ema =ema )
                noise_step = noise_pred_uncond + guidance * (noise_pred- noise_pred_uncond)
            elif guidance==1.0:
                noise_pred = score_function(x_t,cond = c,t=t_vec,ema =ema)
                noise_step = noise_pred
            else:
                noise_pred = score_function(x_t,cond = None,t=t_vec,ema =ema)
                noise_step = noise_pred
     
            if compute_kl:
                noise_pred_old = score_function(x_t,cond = None,t=t_vec,ema =ema,old=True)
                kl+= torch.square(noise_pred_old-noise_step).mean()
           
            x_t,logprob = self.ddim_step_with_logprob(noise_step, x_t, t_vec, t_next_vec, eta,clip_gen ,intervals)
      
        if compute_kl:
                return x_t,kl 
        else:
            return x_t



    def sample_from_model_with_logprob(self,score_function, x0,c,
                                       guidance,num_steps,
                                       randomized=True,
                                       compute_kl=False,
                                       ema=False, 
                                       clip_gen= False,
                                       intervals=[-1,1],
                                       eta=1.0):
        """
        Generate a sample using DDIM.

        Parameters:
        - x_T: Initial noisy sample at timestep torch.
        - num_steps: Number of steps for the sampling schedule.
        - eta: Controls stochasticity; eta=0 is deterministic.

        Returns:
        - x_0: Generated sample at timestep 0.
        """
        # Create a time schedule from T to 0 with `num_steps` intervals
    
        
        sch = DDIMScheduler(num_train_timesteps=self.T, steps_offset=1)
    
        sch.set_timesteps(num_inference_steps=num_steps)

        timesteps = sch.timesteps

        x_t = x0
        #x_traj = [x_t]
        x_traj =[]
        x_traj.append(x_t)
        times = []
        times_next=[]
        ones = torch.ones((x0.size(0),),device=x0.device)
        log_probs=[]
        for i in range(num_steps):
            
            t = timesteps[i]
            t_next = timesteps[i + 1] if i < len(timesteps)-1 else -1
  
            t_vec = (ones * t).long()
            t_next_vec = (ones*t_next).long()
            if guidance!=0.0 and guidance!=1.0:
                #noise_pred, noise_pred_uncond = score_function(x_t,cond =c,t=t_vec,cfg=True)
                
                noise_pred, noise_pred_uncond = score_function(x_t,cond = c,t=t_vec,ema =ema ), score_function(x_t,cond = None,t=t_vec,ema =ema)

                noise_step = noise_pred_uncond + guidance * (noise_pred- noise_pred_uncond)
            elif guidance==1.0:
                noise_step = score_function(x_t,cond = c,t=t_vec,ema=ema)
            else:
                noise_step = score_function(x_t,cond =None,t = t_vec)
                
            noise_step = noise_step.detach()
           
            x_t,logprob = self.ddim_step_with_logprob(noise_step, x_t, t_vec, t_next_vec, eta,clip_gen ,intervals)
            
            assert not torch.isnan(x_t).any(), "x_t contains NaN values"
            assert not torch.isnan(logprob).any(), "logprob contains NaN values"

            times.append(t_vec)
            times_next.append(t_next_vec)
            x_traj.append(x_t)
            log_probs.append(logprob.view(x0.shape[0],1))

        
        if compute_kl:
                return x_t,kl 
        else:
            return x_t



    def _get_variance_logprob(self, timestep, prev_timestep):
        alpha_prod_t = self.alpha_bars_buffer[timestep].to(timestep.device)

        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alpha_bars_buffer[prev_timestep].to(timestep.device) * mask_a
            + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        return variance




    def ddim_step_with_logprob(self,noise_pred,sample,timestep,timestep_prev,eta,clip_gen=False,intervals=[-1,1]):
    
       


        alpha_t = self.alpha_bars_buffer.gather(0, timestep.long()).view(timestep.shape[0], *([1] * (sample.ndim - 1)))
       
        mask_a = (timestep_prev >= 0).int().to(timestep.device)

        mask_b = 1 - mask_a

        alpha_t = self.alpha_bars_buffer[timestep].view(timestep.shape[0], *([1] * (sample.ndim - 1)))
        alpha_t_prev = (
            self.alpha_bars_buffer[timestep_prev].to(timestep.device) * mask_a
            + self.final_alpha_cumprod.to(timestep.device) * mask_b
        ).view(timestep.shape[0], *([1] * (sample.ndim - 1)))

        beta_prod_t = 1 - alpha_t

        variance = self._get_variance_logprob(timestep,timestep_prev).to(dtype=sample.dtype)

        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype).view(timestep.shape[0], *([1] * (sample.ndim - 1)))

    
        pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) 
        pred_original_sample =pred_original_sample/alpha_t ** (0.5)

     
        if clip_gen:
            pred_original_sample = torch.clamp(pred_original_sample,  intervals[0],intervals[1])
            # the model_output is always re-derived from the clipped x_0 in Glide
            noise_pred = (
                sample - alpha_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

 
        pred_sample_direction = (1 - alpha_t_prev - std_dev_t**2) ** (
            0.5
        ) * noise_pred
      

        prev_sample = (alpha_t_prev ** (0.5) * pred_original_sample+ pred_sample_direction)
        log_prob =0
        if eta > 0 and std_dev_t.sum() !=0:
      
            variance_noise = torch.randn(
                    noise_pred.shape,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype)

            variance = std_dev_t * variance_noise
            dist = Normal(prev_sample, std_dev_t)

            prev_sample = prev_sample.detach().clone() + variance
            # log_prob = (
            #         dist.log_prob(prev_sample.detach().clone())
            #         .mean(dim=-1)
            #         .mean(dim=-1)
            #         .mean(dim=-1)
            #         .detach()
            #     )
            if noise_pred.ndim == 4:
                log_prob = (
                    dist.log_prob(prev_sample.detach().clone())
                    .mean(dim=-1)
                    .mean(dim=-1)
                    .mean(dim=-1)
                    .detach()
                )
            else:
                log_prob = (
                    dist.log_prob(prev_sample.detach().clone())
                    .mean(dim=-1)
                    .detach()
                )
      
        return prev_sample,log_prob




    def ddim_step_with_logprob_forward(self,noise_pred,sample,sample_next,timestep,timestep_prev,eta,clip_gen=False,intervals=[-1,1]):
    
        alpha_t = self.alpha_bars_buffer.gather(0, timestep.long()).view(timestep.shape[0], *([1] * (sample.ndim - 1)))
        
        # alpha_prod_t = alpha_prod_t.to(torch.float16)
        mask_a = (timestep_prev >= 0).int().to(timestep.device)

        mask_b = 1 - mask_a

        alpha_t = self.alpha_bars_buffer[timestep].view(timestep.shape[0], *([1] * (sample.ndim - 1)))
        alpha_t_prev = (
            self.alpha_bars_buffer[timestep_prev].to(timestep.device) * mask_a
            + self.final_alpha_cumprod.to(timestep.device) * mask_b
        ).view(timestep.shape[0], *([1] * (sample.ndim - 1)))

        beta_prod_t = 1 - alpha_t

        variance = self._get_variance_logprob(timestep,timestep_prev).to(dtype=sample.dtype)

        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype).view(timestep.shape[0], *([1] * (sample.ndim - 1)))

    
        pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) /alpha_t ** (0.5)
       # pred_original_sample =pred_original_sample/alpha_t ** (0.5)


    
        if clip_gen:
            pred_original_sample = torch.clamp(pred_original_sample, intervals[0],intervals[1] )
            # the model_output is always re-derived from the clipped x_0 in Glide
            noise_pred = (
                sample - alpha_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        pred_sample_direction = (1 - alpha_t_prev - std_dev_t**2) ** (
                0.5
            ) * noise_pred

        prev_sample = (alpha_t_prev ** (0.5) * pred_original_sample+ pred_sample_direction)

        if eta > 0:
        
                variance_noise = torch.randn(
                        noise_pred.shape,
                        device=noise_pred.device,
                        dtype=noise_pred.dtype)

                variance = std_dev_t * variance_noise
                dist = Normal(prev_sample, std_dev_t)

                
                # log_prob = (
                #     dist.log_prob(sample_next.clone().detach())
                #     .mean(dim=-1)
                #     .mean(dim=-1)
                #     .mean(dim=-1)
                # )
                if sample_next.ndim == 4:
                    log_prob = (
                    dist.log_prob(sample_next.clone().detach())
                    .mean(dim=-1)
                    .mean(dim=-1)
                    .mean(dim=-1)
                )
                else:
                    log_prob = (
                    dist.log_prob(sample_next.clone().detach())
                    .mean(dim=-1)
                    )
      
        return prev_sample,log_prob


        # pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_t ** (0.5)

        # if self.clip_gen:
        #     pred_original_sample = 
        # else:
        #     model_output = noise_pred

        # variance = ( (1 - alpha_t_prev)/(1 - alpha_t)) * ( 1 - alpha_t/alpha_t_prev)
        # std_dev_t = eta * variance**(0.5) 
      
        
        # direction = torch.sqrt(1.0 - alpha_t_prev - std_dev_t**2) * model_output
     
      
        # x_t_mean = torch.sqrt(alpha_t_prev) * x_0_pred + direction 
        # variance_n = std_dev_t * torch.randn_like(x_t_mean)
        
        # if sample_prev==None:
        #     prev_sample = x_t_mean + variance_n
        # else:
        #     prev_sample = sample_prev
        
        # log_prob = (
        # -((prev_sample.clone().detach() - x_t_mean) ** 2) / (2 * (std_dev_t**2))
        # - torch.log(std_dev_t)
        # - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))
     
        
        # log_prob = log_prob.mean(dim=list(range(1, prev_sample.ndim)))
   


    def sde(self, t,x=None):
        """
        Returns the drift (f) and diffusion (g) coefficients of the equivalent SDE.

        Parameters:
        - t: Tensor of timesteps.

        Returns:
        - f: Drift coefficient, tensor of shape (batch_size, 1, ...).
        - g: Diffusion coefficient, tensor of shape (batch_size, 1, ...).
        """
        beta_t = self.beta_t(t)
        f = -0.5 * beta_t
        g = torch.sqrt( beta_t)
        if x ==None:
            return  f.view(-1, 1),  g.view(-1, 1)
        else:
            return  f.view(f.shape[0], *([1] * (x.ndim - 1))) ,  g.view(g.shape[0], *([1] * (x.ndim - 1)))




    def mi_sigma(self, s_marg ,s_cond, x_t, g, mean ,std,sigma,importance_sampling):
        M =x_t.size(0)
        N= x_t.size()[1:].numel() 
        chi_t_x = mean**2 * sigma**2 + std**2
        ref_score_x = -(x_t)/chi_t_x 
        const = 0 #get_normalizing_constant((1,)).to(s_marg.device)
        if importance_sampling:
            e_marg =  -const * 0.5 * ((s_marg - std* ref_score_x)**2)
            e_cond =  -const * 0.5 * ((s_cond - std* ref_score_x)**2)
        else:
            e_marg =  -0.5*((g/std)**2* ((s_marg -  std *ref_score_x)**2) )
            e_cond =  -0.5*( (g/std)**2* ((s_cond -  std * ref_score_x)**2 ) )
        return (e_marg - e_cond ).sum(dim=list(range(1, x_t.ndim)))



    def mi_theoritic(self, s_marg ,s_cond, z, g, mean ,std,sigma,importance_sampling=False):

        ref_score_x = -z
        const = 0# get_normalizing_constant((1,)).to(s_marg.device)
        if importance_sampling:
            e_marg =  const * 0.5 * ((s_marg -  ref_score_x)**2)
            e_cond =  const * 0.5 * ((s_cond - ref_score_x)**2)
        else:
            e_marg = 0.5*((g/std)**2* ((s_marg -   ref_score_x )**2) )
            e_cond = 0.5*((g/std)**2* ((s_cond -   ref_score_x)**2 ) )
        return (e_marg - e_cond).sum(dim=list(range(1, z.ndim)))


    def mmse(self,x0,x_t, eps_hat ,mean,std):
        x0_hat = (x_t - std * eps_hat )/mean
        return torch.sum((x0 - x0_hat)**2,dim=list(range(1, x0.ndim)))

    def entropy_theoritic(self, s, z, g, mean ,std,sigma,importance_sampling=False,entropy_weight=False):
        ref_score_x = -z
        const = 0 #get_normalizing_constant((1,)).to(s.device)
        if importance_sampling:
            e_cond =  const * 0.5 * (( s -  ref_score_x)**2)
        else:
            if entropy_weight:
                e_cond = 0.5*((g/std)**2* ((s -   ref_score_x)**2 ) )
            else:
                e_cond =  (s -   ref_score_x)**2  
    
        return e_cond.sum(dim=list(range(1, z.ndim)))


    def entropy(self,  s,x_0, x_t, g, mean ,std,sigma,importance_sampling):

        M =x_t.size(0)
        N= x_t.size()[1:].numel()

        chi_t_x = mean**2 * sigma**2 + std**2
        ref_score_x = -(x_t)/chi_t_x 

        term = N*0.5*np.log(2 *np.pi ) + 0.5* torch.sum(x_0**2)/M #- 0.5 * N * torch.sum( torch.log(chi_t_x) -1 +  1 / chi_t_x ) 

        const = 0#get_normalizing_constant((1,)).to(s.device)
        if importance_sampling:
            e_cond = term - const * 0.5 * (( s - std * ref_score_x)**2).sum(dim=list(range(0, x_0.ndim))) /M
        else:
            e_cond = term -0.5*( (g/std)**2* (( s -  std * ref_score_x)**2 ) ).sum(dim=list(range(0, x_0.ndim))) /M
        return e_cond


    def mi(self, s_marg ,s_cond,std, g, importance_sampling=False):

        M = g.shape[0] 

        const = 0 #get_normalizing_constant((1,)).to(s_marg.device)
        
        if importance_sampling:
            mi = const *0.5* ((s_marg - s_cond  )**2) 
        else:
            mi = 0.5* ( (g/std)**2*(s_marg - s_cond)**2)
        
        return mi


   
    def snr_t(self, t: int, normalize: bool = True) -> torch.Tensor:

        eps = 1e-5

        # SNR and log-SNR computation
        alpha_bar_t = self.alpha_bars_buffer[t]  # scalar tensor
        snr_t = alpha_bar_t / (1. - alpha_bar_t + eps)
        log_snr_t = torch.log(snr_t + eps)

        if normalize:
            snr_all = self.alpha_bars_buffer / (1. - self.alpha_bars_buffer + eps)
            print(snr_all)
            log_snr_all = torch.log(snr_all + eps)
            log_snr_t = log_snr_t / log_snr_all.sum()
        return log_snr_t
    

    def select_timesteps(self, n=20, low_frac=0.05, high_frac=0.95, sampling="linear"):
        T = self.T
        eps = 1e-5

        # Ensure bounds are within valid range
        low_bound = max(int(low_frac * T), 0)
        high_bound = min(int(high_frac * T), T - 1)

        if sampling == "linear":
            steps = torch.linspace(low_bound, high_bound, n, device=self.device).long()
        elif sampling == "log":
            log_start = torch.log10(torch.tensor(float(low_bound)))
            log_end = torch.log10(torch.tensor(float(high_bound)))
            log_steps = torch.linspace(log_start, log_end, steps=n)
            steps = torch.unique(torch.round(10 ** log_steps).long())
        elif sampling == "uniform":
            steps = torch.randint(low_bound, high_bound + 1, (n,), device=self.device)
        elif sampling == "logit":
            # Normalize to (0, 1) before applying logit
            norm_vals = torch.linspace(0.01, 0.99, n, device=self.device)
            steps = torch.sigmoid(torch.logit(norm_vals)).mul(high_bound - low_bound).add(low_bound).long()
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")

        snr = self.alpha_bars_buffer / (1. - self.alpha_bars_buffer + eps)
        snr = snr[steps]

        # Normalize weights
        weights = torch.log(snr + eps)
        weights = weights - weights.min()
        weights = weights / weights.sum()

        return steps.to(self.device), weights.to(self.device) * n
    

    
    def t2logsnrs(self):
        """The logsnr values that were trained, according to the scheduler."""
        logsnrs_trained = torch.log(self.alpha_bars_buffer) - torch.log1p(-self.alpha_bars_buffer) # Logit function is inverse of sigmoid
        return logsnrs_trained


    def logsnr2t(self, logsnr):
        """Directly use alphas_cumprod schedule from scheduler object"""
        logsnrs_trained = self.t2logsnrs().to(logsnr.device)
        assert len(logsnr.shape) == 1, "not a 1-d tensor"
        timestep = torch.argmin(torch.abs(logsnr.view(-1, 1) - logsnrs_trained), dim=1)
        return timestep

    def convert_t_logsnr(self,t):
        log_snr = torch.log(self.alpha_bars_buffer[t]) - torch.log1p(-self.alpha_bars_buffer[t]) # Logit function is inverse of sigmoid
        return log_snr


    def training_losses(self,model, x_start, t,model_kwargs=None):
        """Compute the training loss for a batch of data at a given timestep."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        x_t, z, mean, std = self.q_sample(x_start, t)
        noise_pred = model(x_t, t=t, **model_kwargs)
        loss =  ((noise_pred - z) ** 2).mean(dim=list(range(1, z.ndim)))
        return {"loss": loss}



    def nll_grad(self,x,cond,model, t=None,w=None,reduce="mean", z_per_samples_per_x = 1,one_step=True , type="eps"):
 
        mse = 0
        d = x.shape[1:].numel()
        for i in range(t.shape[0]):
            if one_step:
                weight = w[i].repeat(x.shape[0]).view(x.shape[0], *([1] * (x.ndim - 1)))
                t_x = t[i].repeat(x.shape[0])
            else:
                weight = w[i].view(-1,*list(range(1, x.ndim)))
                t_x = t[i]
      
            for _ in range(z_per_samples_per_x):
           
                    x_t, z, mean, std = self.q_sample(x,t_x)
                    f,g = self.sde(t_x,x_t)
                    #ww = 0.5 *(g/std)**2
                    ww = 0.5
                    eps_cond =  model(x_t,cond=cond,t=t_x, ema=False)
                    if type=="x":
                        x_0_hat = (x_t - std * eps_cond )/mean
                        mse += 0.5* weight * ((x-x_0_hat)**2)/ (z_per_samples_per_x*len(t))
                    else:
                        #mse += 0.5 *weight*((z-eps_cond)**2)/ (z_per_samples_per_x*len(t))
                        mse += ww*((z-eps_cond)**2)/ (z_per_samples_per_x*len(t))

             
        if reduce =="mean":
            return mse.sum(dim=list(range(1, z.ndim))) /d
        else:
            return mse.sum(dim=list(range(1, z.ndim))) 
  
        
    def nll(self,x,cond,model, t=None,w=None,reduce="mean", z_per_samples_per_x = 1 , loss_type="mse" , weighted = False, sampling="linear"):
    
      
        t,weights = self.select_timesteps(n=len(t),sampling=sampling)
        mses = 0
        for i in range(len(t)):
            t_x = t[i].repeat(x.shape[0])
            ww = weights[i].repeat(x.shape[0]).view(x.shape[0], *([1] * (x.ndim - 1)))

            for _ in range(z_per_samples_per_x):
                x_t, z, mean, std = self.q_sample(x,t_x)

                with torch.no_grad():
                        eps_cond =  model(x_t,cond=cond,t=t_x, ema=False).detach()
                if loss_type=="mse":
                        mse =  ((eps_cond-z)**2)/ (z_per_samples_per_x*len(t))
                elif loss_type == "kl":
                    # Use internal buffers
                    B = x.shape[0]
                    alpha_t = self.alpha_t(t_x).view(-1,1)
                    alpha_bar_t = self.alpha_bar_t(t_x).view(-1,1)
                    beta_t = self.beta_t(t_x).view(-1,1)

                    
                    # True mean (uses real noise z)
                    true_mean = (1. / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1. - alpha_bar_t)) * z)

                    # Model mean (uses predicted noise)
                    model_mean = (1. / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1. - alpha_bar_t)) * eps_cond)

                    # KL divergence (up to constants)
                    mse = ((true_mean - model_mean) ** 2) / beta_t
                   
                    mse = mse / (z_per_samples_per_x * len(t))
                elif loss_type=="mi":
                        eps_uncond =  model(x_t,cond=None,t=t_x, old=True).detach()
                        mse = -  ((eps_uncond-z)**2 - (eps_cond-z)**2)/ (z_per_samples_per_x*len(t))
                if weighted:
                        mse *= ww
                mses += mse
        if reduce=="sum":
            return mses.sum(dim=list(range(1, z.ndim)))
        elif reduce=="mean":
            return mses.mean(dim=list(range(1, z.ndim))) 
       
        return mses 
        


   

def logistic_integrate(npoints, loc, scale, clip=4., device='cpu', deterministic=False):
    """Return sample point and weights for integration, using
    a truncated logistic distribution as the base, and importance weights.
    """
    loc, scale, clip = torch.tensor(loc, device=device), torch.tensor(scale, device=device), torch.tensor(clip, device=device)

    # IID samples from uniform, use inverse CDF to transform to tar_prompt distribution
    if deterministic:
        torch.manual_seed(0)
    ps = torch.rand(npoints, dtype=loc.dtype, device=device)
    ps = torch.sigmoid(-clip) + (torch.sigmoid(clip) - torch.sigmoid(-clip)) * ps  # Scale quantiles to clip
    logsnr = loc + scale * torch.logit(ps)  # Using quantile function for logistic distribution

    # importance weights
    weights = scale * torch.tanh(clip / 2) / (torch.sigmoid((logsnr - loc)/scale) * torch.sigmoid(-(logsnr - loc)/scale))
    return logsnr, weights


def logistic_integrate(npoints, loc=1., scale=2., clip=3., device='cpu', deterministic=False):
    """Return sample point and weights for integration, using
    a truncated logistic distribution as the base, and importance weights.
    """
    loc, scale, clip = torch.tensor(loc, device=device), torch.tensor(scale, device=device), torch.tensor(clip, device=device)

    # IID samples from uniform, use inverse CDF to transform to tar_prompt distribution
    if deterministic:
        torch.manual_seed(0)
    ps = torch.rand(npoints, dtype=loc.dtype, device=device)
    ps = torch.sigmoid(-clip) + (torch.sigmoid(clip) - torch.sigmoid(-clip)) * ps  # Scale quantiles to clip
    logsnr = loc + scale * torch.logit(ps)  # Using quantile function for logistic distribution

    # importance weights
    weights = scale * torch.tanh(clip / 2) / (torch.sigmoid((logsnr - loc)/scale) * torch.sigmoid(-(logsnr - loc)/scale))
    return logsnr, weights


def one_step_test(loc, scale, clip=4., device='cpu'):
    logsnr = torch.tensor([loc + clip * scale], device=device)
    loc, scale, clip = torch.tensor(loc, device=device), torch.tensor(scale, device=device), torch.tensor(clip, device=device)
    weight = scale * torch.tanh(clip / 2) / (torch.sigmoid((logsnr - loc) / scale) * torch.sigmoid(-(logsnr - loc) / scale))
    return logsnr, weight



def trunc_uniform_integrate(npoints, loc, scale, clip=4., device='cpu', deterministic=False):
    """Return sample point and weights for integration, using
    a truncated distribution proportional to 1 / (1+snr) as the base, and importance weights.
    loc, scale, clip  - are same as for continuous density estimator, just used to fix the range
    parameter, eps=1 is the form implied by optimal Gaussian MMSE at low SNR.
    True MMSE drops faster, so we use a smaller constant
    """
    loc, scale, clip =torch.tensor(loc, device=device),torch.tensor(scale, device=device),torch.tensor(clip, device=device)
    left_logsnr, right_logsnr = loc - clip * scale, loc + clip * scale  # truncated range

    # IID samples from uniform, use inverse CDF to transform to target distribution
    if deterministic:
       torch.manual_seed(0)
    ps =torch.rand(npoints, dtype=loc.dtype, device=device)
    logsnrs = left_logsnr + (right_logsnr - left_logsnr) * ps  # Use quantile function

    # importance weights
    weights = torch.ones(npoints, device=device) / (right_logsnr - left_logsnr)
    return logsnrs, weights


def soft_round(x, snr, xinterval, delta):
    ndim = len(x.shape)
    bins = torch.linspace(xinterval[0], xinterval[1], 1 + int((xinterval[1]-xinterval[0])/delta), device=x.device)
    bins = bins.reshape((-1,) + (1,) * ndim)
    ps = torch.nn.functional.softmax(-0.5 * torch.square(x - bins) * (1 + snr), dim=0)
    return (bins * ps).sum(dim=0)
