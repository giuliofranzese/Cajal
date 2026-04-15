
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler
from torch.distributions import Normal


class DDPM(nn.Module):
    def __init__(
        self,
        beta_schedule: str = "linear",
        num_train_timesteps: int = 1000,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        # Diffusers schedulers (no .to() because these are not nn.Modules)
        self.ddpm_scheduler = DDPMScheduler(
            beta_schedule=beta_schedule,
            num_train_timesteps=num_train_timesteps,
        )
        self.ddim_scheduler = DDIMScheduler(
            steps_offset=1,
            beta_schedule=beta_schedule,num_train_timesteps =num_train_timesteps,
            clip_sample=False,set_alpha_to_one=False
        )

    def set_device(self, device):
        self.device = device
        # No .to() on schedulers; device management is handled during sampling

    def beta_t(self, t):
        return self.ddpm_scheduler.betas[t.cpu()]

    def alpha_t(self, t):
        return self.ddpm_scheduler.alphas[t.cpu()]

    def alpha_bar_t(self, t):
        return self.ddpm_scheduler.alphas_cumprod[t.cpu()]

    def marg_prob(self, t, x):
        mean = torch.sqrt(self.alpha_bar_t(t)).to(self.device)
        std = torch.sqrt(1 - self.alpha_bar_t(t)).to(self.device)
        mean = mean.view(-1, *([1] * (x.ndim - 1)))
        std = std.view(-1, *([1] * (x.ndim - 1)))
        return mean, std

    def q_sample(self, x0, t):
        # forward diffusion using DDPM scheduler
        noise = torch.randn_like(x0, device=self.device)
        x_t = self.ddpm_scheduler.add_noise(x0, noise, t)
        mean, std = self.marg_prob(t, x0)
        return x_t, noise, mean, std

    def sample_uniform(self, shape, eps=0.0, max_eps=None):
        T = self.ddpm_scheduler.config.num_train_timesteps
        eps_steps = int(T * eps)
        max_eps_steps = eps_steps if max_eps is None else int(T * max_eps)
        return torch.randint(
            low=eps_steps,
            high=T - max_eps_steps,
            size=shape,
            device=self.device
        )

    def sample_from_model(
        self,
        score_function,
        x0,
        cond,
        guidance,
        num_steps,
        clip_gen=False,
        compute_kl=False,
        ema=False,
        eta=0.0,
        intervals=[-1, 1]
    ):
        # prepare DDIM timesteps
        self.ddim_scheduler.set_timesteps(num_inference_steps=num_steps)
        timesteps = self.ddim_scheduler.timesteps
        x_t = x0.to(self.device)
        ones = torch.ones((x0.size(0),), device=self.device, dtype=torch.long)
        kl = 0.0
        for i, t in enumerate(timesteps):
            t_vec = ones * t

            # Predict noise with guidance
            if 0 < guidance < 1 and cond is not None:
                eps = score_function(x_t, cond=cond, t=t_vec.to(self.device), ema=ema)
                eps_uncond = score_function(x_t, cond=None, t=t_vec.to(self.device), ema=ema)
                noise_pred = eps_uncond + guidance * (eps - eps_uncond)
            else:
                ctx = cond if guidance == 1.0 else None
                noise_pred = score_function(x_t, cond=ctx, t=t_vec.to(self.device), ema=ema)

            if compute_kl:
                eps_old = score_function(x_t, cond=None, t=t_vec.to(self.device), ema=ema, old=True)
                kl += torch.square(eps_old - noise_pred).mean()

            # DDIM step
            out = self.ddim_scheduler.step(
                model_output=noise_pred,
                timestep=int(t),
                sample=x_t,
                eta=eta,
            )
            
            x_t = out.prev_sample.to(self.device)

        return (x_t, kl) if compute_kl else x_t

    def sample_from_model_with_logprob(
        self,
        score_function,
        x0,
        cond,
        guidance,
        num_steps,
        compute_kl=False,
        ema=False,
        clip_gen=False,
        intervals=[-1, 1],
        eta=1.0
    ):
        #timesteps = self.ddim_scheduler.set_timesteps(num_inference_steps=num_steps)
        self.ddim_scheduler.set_timesteps(num_inference_steps=num_steps)
        timesteps = self.ddim_scheduler.timesteps

        x_t = x0.to(self.device)
        ones = torch.ones((x0.size(0),), device=self.device, dtype=torch.long)

        traj, log_probs, times, times_next = [x_t], [], [], []
        for i, t in enumerate(timesteps):
            t_vec = ones * t
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else -1
            t_next_vec = ones * t_next
            # guidance
            if guidance!=0 and guidance!=1 and cond is not None:
                eps = score_function(x_t, cond=cond, t=t_vec.to(self.device), ema=ema)
                eps_uncond = score_function(x_t, cond=None, t=t_vec.to(self.device), ema=ema)
                noise_pred = eps_uncond + guidance * (eps - eps_uncond)
            else:
                ctx = cond if guidance == 1.0 else None
                noise_pred = score_function(x_t, cond=ctx, t=t_vec.to(self.device), ema=ema)

    
            prev, lp = self.step_forward_logprob(model_output=noise_pred,timestep=t,sample=x_t,next_sample=None,eta=eta)
            
          


            traj.append(prev)
            log_probs.append(lp.view(x0.shape[0],1))
            times.append(t_vec)
            times_next.append(t_next_vec)
        
        print("traj",len(traj))

        i,j = 25,25 
        traj_test = traj[:-1]
        traj_prev = traj[1:]
        print("shapes",traj_test[i].shape,traj_test[j].shape)
        x_t = torch.cat([traj_test[i][:3,:],traj_test[j][:3,:]],dim=0)
        x_t_next = torch.cat([traj_prev[i][:3,:],traj_prev[j][:3,:]],dim=0)
        t = torch.cat([times[i][:3],times[j][:3]])
        t_next = torch.cat([times_next[i][:3],times_next[j][:3]])

        ones = torch.ones((x_t_next.size(0),), device=self.device, dtype=torch.long)
        t_vec = ones * t
        t_next_vec = ones * t_next
        cond_test = torch.cat([cond[:3],cond[:3]],dim=0)
      

        if guidance!=0 and guidance!=1 and cond is not None:
            eps = score_function(x_t, cond=cond_test, t=t_vec.to(self.device), ema=ema)
            eps_uncond = score_function(x_t, cond=None, t=t_vec.to(self.device), ema=ema)
            noise_pred_ = eps_uncond + guidance * (eps - eps_uncond)
        else:
            ctx = cond if guidance == 1.0 else None
            noise_pred = score_function(x_t, cond=ctx, t=t_vec.to(self.device), ema=ema)
            
        prev_, lp_ = self.step_forward_logprob(model_output=noise_pred_,timestep=t,sample=x_t,next_sample=x_t_next,eta=eta)
        print("==============")
        print("lp_",lp_)
        print("lp_old",log_probs[i][:3].squeeze(),log_probs[j][:3].squeeze())
        print("t",t)
        print("t_next_vec",t_next)
        print("==============")
        return traj, log_probs, times, times_next
    
    
    
    def step_forward_logprob(
      self_,
      model_output,
      timestep,
      sample,
      next_sample,
      eta = 1.0,
      use_clipped_model_output = False,
      generator=None,
      variance_noise = None,
      return_dict = True,
  ):  # pylint: disable=g-bare-generic
        """Predict the sample at the previous timestep by reversing the SDE.

        Core function to propagate the diffusion process from the learned model
        outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion
            model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`): current instance of sample (x_t) being
            created by diffusion process.
            next_sample (`torch.FloatTensor`): instance of next sample (x_t-1) being
            created by diffusion process. RL sampling is the backward process,
            therefore, x_t-1 is the "next" sample of x_t.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected"
            `model_output` from the clipped predicted original sample. Necessary
            because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened,
            "corrected" `model_output` would coincide with the one provided as
            input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for
            the variance using `generator`, we can directly provide the noise for
            the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than
            DDIMSchedulerOutput class

        Returns:
            log probability.
        """
        self = self_.ddim_scheduler
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps'"
                " after creating the scheduler"
            )
        timestep = timestep.cpu()
    
        # pylint: disable=line-too-long
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
            + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t
        variance = self_._get_variance_logprob(timestep, prev_timestep).to(
            dtype=sample.dtype
        )
        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)

   

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf

        # beta_prod_t = beta_prod_t.view(-1,list(range(1,model_output.ndim)))
        # alpha_prod_t = alpha_prod_t.view(-1,list(range(1,model_output.ndim)))
        if len(timestep.shape) > 0:  # If timesteps has multiple values
            beta_prod_t = beta_prod_t.view(-1, *[1]*(model_output.ndim-1)).to(sample.device)
            alpha_prod_t = alpha_prod_t.view(-1, *[1]*(model_output.ndim-1)).to(sample.device)
            alpha_prod_t_prev = alpha_prod_t_prev.view(-1, *[1]*(model_output.ndim-1)).to(sample.device)
            std_dev_t = std_dev_t.view(-1, *[1]*(model_output.ndim-1)).to(sample.device)
            variance = variance.view(-1, *[1]*(model_output.ndim-1)).to(sample.device)

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one"
                " of `epsilon`, `sample`, or `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # pylint: disable=line-too-long
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * model_output

        # pylint: disable=line-too-long
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample
            + pred_sample_direction
        )

        if eta > 0:
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure"
                    " that either `generator` or `variance_noise` stays `None`."
                )

        if variance_noise is None:

            variance_noise = torch.randn_like(model_output)
        

        std_dev_t = std_dev_t.to(self_.device)
        
    
        noise = std_dev_t * variance_noise
        dist = Normal(prev_sample, std_dev_t)
       
        if next_sample is None:
            next_sample = prev_sample + noise
            
        log_prob = (
                dist.log_prob(next_sample.detach().clone()).mean(dim=list(range(1,next_sample.ndim)))
            )

        
        return next_sample, log_prob
    


    def _get_variance_logprob(self_, timestep, prev_timestep):
        self = self_.ddim_scheduler
     
        alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
            + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance
    
    def sde(self, t, x=None):
        beta = self.beta_t(t)
        f = -0.5 * beta
        g = torch.sqrt(beta)
        shape = (-1, *([1] * (x.ndim - 1))) if x is not None else (-1, 1)
        return f.to(self.device).view(*shape), g.to(self.device).view(*shape)

    def nll(self, x, cond, model, t, w, one_step=True, type="eps",reduce = "mean"):
    
        #t = self.ddim_scheduler.timesteps
        
        mse = 0.0
