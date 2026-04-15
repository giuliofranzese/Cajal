
import torch
import itertools
import numpy as np



class VP_SDE():
    def __init__(self,
                 beta_min=0.1,
                 beta_max=20,
                 N=1000,
                 importance_sampling=True,
                 ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.rand_batch = True
        self.N = N
        self.T = 1
        self.importance_sampling = importance_sampling
        self.device = "cuda"

       

    def set_device(self, device):
        self.device = device

    def sample_uniform(self, shape,eps = 1e-5):
        return ((1- eps) *
                 torch.rand((shape, 1)) + eps).to(self.device)
    
    def sample_importance_sampling_t(self, shape):
        """
            Non-uniform sampling of t to importance_sampling. See [1,2] for more details.
            [1] https://arxiv.org/abs/2106.02808
            [2] https://github.com/CW-Huang/sdeflow-light
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, T=self.T)

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, t, x=None):
        # Returns the drift and diffusion coefficient of the SDE ( f(t), g(t)) respectively.
        f = -0.5*self.beta_t(t)
        g = torch.sqrt(self.beta_t(t))
        if x==None:
            return f.view(-1,1),g.view(-1,1)
        else:
            return f.view(f.shape[0], *([1] * (x.ndim - 1))) , g.view(f.shape[0], *([1] * (x.ndim - 1)))

    def marg_prob(self, t, x):
        
        ## Returns mean and std of the marginal distribution P_t(x_t) at time t.
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))

        mean = mean.view(mean.shape[0], *([1] * (x.ndim - 1)))
        std = std.view(std.shape[0], *([1] * (x.ndim - 1)))
        return mean , std

    def sample(self, x_0, t):
        ## Forward SDE
        # Sample from P(x_t | x_0) at time t. Returns A noisy version of x_0.
        mean, std = self.marg_prob(t, x_0)

        z = torch.randn_like(x_0, device=self.device)
        x_t = mean * x_0  + std * z
        return x_t, z, mean, std

   

    def sample_from_model(self,score_function, x0,c, guidance,num_steps,randomized=True,compute_kl=False,ema=False):
        ## Euler solver
        if randomized:
            num_steps_ = (num_steps +torch.randint(0,10,(1,),device=x0.device).cpu().item()) +torch.randn((1,),device=x0.device) 
            dt = (self.T) / num_steps_
            t = (self.T -  torch.empty(1).uniform_(0, dt.item()) ).to(x0.device)
        else:
            dt = torch.tensor( [self.T / num_steps], device=x0.device)
            t = self.T
        t_vec = torch.ones((x0.size(0),1),device=x0.device)


        print("hihi")
        kl =0
        k =0
       # mean,std = self.marg_prob(t * t_vec,x0)

        #xt = x0 * mean + std * torch.randn_like(x0)
        xt = x0
        while t>=1e-5:
            t_tens =  t * t_vec 
            f,g = self.sde(t_tens,x0)

            _,std = self.marg_prob(t_tens,xt)

            score =  score_function(xt,c, t_tens,ema=ema)
            if guidance!=0:
                score_uncond = score_function(xt,None,t_tens,ema=ema) 
                score_step = score_uncond + guidance * (score -score_uncond )
            else:
                score_step = score

            score_step = -score_step/std
            if t > 1e-5+dt:
                noise = torch.randn_like(xt)
            else:
                noise = 0

            reverse_mean = xt - dt*(f*xt - (g**2) *score_step)
            x_t_prev = reverse_mean  + torch.sqrt(dt) * g * noise

            if compute_kl:
                old_score =  -score_function(xt,c, t_tens,old=True,ema=ema)/std
                old_score_step = score_uncond + guidance * (old_score -score_uncond )
                reverse_mean_old = xt - dt*(f*xt - (g**2) *old_score_step)
                kl += ( 0.5 * g**2 * torch.square(reverse_mean-reverse_mean_old) ).mean() 
            k+=1
            t -= dt
            xt = x_t_prev
        return xt 
    
    



def mi_sigma( s_marg ,s_cond, x_t, g, mean ,std,sigma,importance_sampling):
    
    M =x_t.size(0)
    N= x_t.size()[1:].numel() 
    chi_t_x = mean**2 * sigma**2 + std**2
    ref_score_x = -(x_t)/chi_t_x 
    const = get_normalizing_constant((1,)).to(s_marg.device)
    if importance_sampling:
        e_marg =  -const * 0.5 * ((s_marg - std* ref_score_x)**2)
        e_cond =  -const * 0.5 * ((s_cond - std* ref_score_x)**2)
    else:
        e_marg =  -0.5*((g/std)**2* ((s_marg -  std *ref_score_x)**2) )
        e_cond =  -0.5*( (g/std)**2* ((s_cond -  std * ref_score_x)**2 ) )
    return (e_marg - e_cond ).sum(dim=list(range(1, x_t.ndim)))



def mi_theoritic( s_marg ,s_cond, z, g, mean ,std,sigma,importance_sampling):

    ref_score_x = -z
    const = get_normalizing_constant((1,)).to(s_marg.device)
    if importance_sampling:
        e_marg =  const * 0.5 * ((s_marg -  ref_score_x)**2)
        e_cond =  const * 0.5 * ((s_cond - ref_score_x)**2)
    else:
        e_marg = 0.5*((g/std)**2* ((s_marg -   ref_score_x )**2) )
        e_cond = 0.5*((g/std)**2* ((s_cond -   ref_score_x)**2 ) )
    return (e_marg - e_cond).sum(dim=list(range(1, z.ndim)))



def entropy_theoritic( s, z, g, mean ,std,sigma,importance_sampling):
    ref_score_x = -z
    const = get_normalizing_constant((1,)).to(s.device)
    if importance_sampling:
        e_cond =  const * 0.5 * (( s -  ref_score_x)**2)
    else:
        e_cond = 0.5*((g/std)**2* ((s -   ref_score_x)**2 ) )

   
    return e_cond.sum(dim=list(range(1, z.ndim)))



def entropy(  s,x_0, x_t, g, mean ,std,sigma,importance_sampling):

    M =x_t.size(0)
    N= x_t.size()[1:].numel()

    chi_t_x = mean**2 * sigma**2 + std**2
    ref_score_x = -(x_t)/chi_t_x 

    term = N*0.5*np.log(2 *np.pi ) + 0.5* torch.sum(x_0**2)/M #- 0.5 * N * torch.sum( torch.log(chi_t_x) -1 +  1 / chi_t_x ) 

    const = get_normalizing_constant((1,)).to(s.device)
    if importance_sampling:
        e_cond = term - const * 0.5 * (( s - std * ref_score_x)**2).sum(dim=list(range(0, x_0.ndim))) /M
    else:
        e_cond = term -0.5*( (g/std)**2* (( s -  std * ref_score_x)**2 ) ).sum(dim=list(range(0, x_0.ndim))) /M
    return e_cond



def mi( s_marg ,s_cond,std, g, importance_sampling):

    M = g.shape[0] 

    const = get_normalizing_constant((1,)).to(s_marg.device)
    
    if importance_sampling:
        mi = const *0.5* ((s_marg - s_cond  )**2) 
    else:
        mi = 0.5* (g**2*(s_marg/std - s_cond/std )**2)
        
    return mi


import torch
import numpy as np

""" Code reported from :
        [1] https://arxiv.org/abs/2106.02808
        [2] https://github.com/CW-Huang/sdeflow-light
"""
t_eps= 1e-3


def sample_vp_truncated_q(shape, beta_min, beta_max, T,t_epsilon=1e-3):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(beta_min=beta_min, beta_max=beta_max, t_epsilon=t_epsilon)
    return vpsde.inv_Phi(u.view(-1), T).view(*shape)


# noinspection PyUnusedLocal
def get_normalizing_constant(shape,T =1.0):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(beta_min=0.1, beta_max=20.0, t_epsilon=0.001)
    return vpsde.normalizing_constant(T=T)



class VariancePreservingTruncatedSampling:

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20., t_epsilon=1e-3):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_max - self.beta_min) + t * self.beta_min

    def mean_weight(self, t):
        # return torch.exp( -0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min )
        return torch.exp(-0.5 * self.integral_beta(t))

    def var(self, t):
        # return 1. - torch.exp( -0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min )
        return 1. - torch.exp(- self.integral_beta(t))

    def std(self, t):
        return self.var(t) ** 0.5

    def g(self, t):
        beta_t = self.beta(t)
        return beta_t ** 0.5

    def r(self, t):
        return self.beta(t) / self.var(t)

    def t_new(self, t):
        mask_le_t_eps = (t <= self.t_epsilon).float()
        t_new = mask_le_t_eps * t_eps + (1. - mask_le_t_eps) * t
        return t_new

    def unpdf(self, t):
        t_new = self.t_new(t)
        unprob = self.r(t_new)
        return unprob

    def antiderivative(self, t):
        return torch.log(1. - torch.exp(- self.integral_beta(t))) + self.integral_beta(t)

    def phi_t_le_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.r(t_eps).item() * t

    def phi_t_gt_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.phi_t_le_t_eps(t_eps).item() + self.antiderivative(t) - self.antiderivative(t_eps).item()

    def normalizing_constant(self, T):
        return self.phi_t_gt_t_eps(T)



Log2PI = float(np.log(2 * np.pi))

def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z

def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1

def sample_gaussian(shape):
    return torch.randn(*shape)

def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')
