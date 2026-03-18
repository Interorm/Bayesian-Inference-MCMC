import torch
from torch.special import gammaln
import torch.distributions as dist

import tqdm.auto as tqdm 
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, clear_output
import numpy as np


def propose_beta(beta:torch.Tensor, k:torch.Tensor):

    CHAINS, P, R, C = beta.shape
    concentration = (beta * k[:, None, None, None]).reshape(CHAINS, P * R, C)

    beta_prop = dist.Dirichlet( concentration ).sample()

    return beta_prop.reshape(CHAINS, P, R, C)


def acceptance_beta(beta:torch.Tensor, beta_prop:torch.Tensor,
                    alpha:torch.Tensor, X:torch.Tensor, T:torch.Tensor, k:torch.Tensor):
    
    k = k.clone()[:, None, None, None]

    beta_log = torch.log(beta)
    beta_prop_log = torch.log(beta_prop)

    log_prior = (alpha.unsqueeze(1).exp() * (beta_prop_log - beta_log))

    theta = (X[None, :, :, None] * beta).sum(dim=2)
    theta_prop = theta[:, :, None, :] + (X[None, :, :, None] * (beta_prop - beta))
    log_likelihood = T[None, :, None, :] * (torch.log(theta_prop) - torch.log(theta[: ,:, None, :]))

    log_proposal = (k * beta_prop - 1) * beta_log - (k * beta - 1) * beta_prop_log + \
                   gammaln(k * beta) - gammaln(k * beta_prop)
    

    log_acceptance = (log_prior + log_likelihood + log_proposal).sum(dim=3)

    return log_acceptance


def beta_step(beta:torch.Tensor, k:torch.Tensor,
              alpha:torch.Tensor, X:torch.Tensor, T:torch.Tensor):
    
    beta_prop = propose_beta(beta, k)

    acceptance = acceptance_beta(beta, beta_prop, alpha, X, T, k)

    acceptance_threshhold = torch.log(dist.Uniform(torch.zeros_like(acceptance),
                                                   torch.ones_like(acceptance)).sample())
    acceptance_mask:torch.Tensor = acceptance_threshhold <= acceptance 
    accept_rate = acceptance_mask.float().mean(dim=(1,2)).cpu()

    acceptance_mask = acceptance_mask.unsqueeze(-1).expand_as(beta)
    beta_new = torch.where(acceptance_mask, beta_prop, beta)
    
    return beta_new, accept_rate



def propose_alpha(alpha:torch.Tensor, sigma:torch.Tensor):

    alpha_prop = dist.Normal(
                        loc = alpha,
                        scale = sigma[:, None, None].expand_as(alpha)
                    ).sample()
    
    return alpha_prop


def acceptance_alpha(alpha:torch.Tensor, alpha_prop:torch.Tensor, beta:torch.Tensor, alpha_prior:tuple):

    CHAINS, P, R, C = beta.shape
    alpha_prior_mu, alpha_prior_sigma = alpha_prior

    alpha = alpha.exp()
    alpha_prop = alpha_prop.exp()

    alpha_expandedRCC = alpha.unsqueeze(2).expand(CHAINS, R, C, C).clone()
    torch.diagonal(alpha_expandedRCC, dim1=2, dim2=3).copy_(alpha_prop)


    log_joined_gamma = P * (gammaln(alpha_expandedRCC.sum(dim=3)) - gammaln(alpha.sum(dim=2))[:, :, None])

    log_indiv_gamma = P * (gammaln(alpha) - gammaln(alpha_prop))
    
    log_kernel = (alpha_prop - alpha) * torch.log(beta).sum(dim=1)


    log_prior = 0.5 / alpha_prior_sigma**2 * ( (alpha - alpha_prior_mu)**2 - (alpha_prop - alpha_prior_mu)**2 )


    log_acceptance = log_joined_gamma + log_indiv_gamma + log_kernel + log_prior

    return log_acceptance


def alpha_step(alpha:torch.Tensor, sigma:torch.Tensor, beta:torch.Tensor, alpha_prior:tuple):

    alpha_prop = propose_alpha(alpha, sigma)

    acceptance = acceptance_alpha(alpha, alpha_prop, beta, alpha_prior)

    acceptance_threshhold = torch.log(dist.Uniform(torch.zeros_like(acceptance),
                                                   torch.ones_like(acceptance)).sample())
    acceptance_mask = acceptance_threshhold <= acceptance 
    accept_rate = acceptance_mask.float().mean(dim=(1,2)).cpu()
    alpha_new = torch.where(acceptance_mask, alpha_prop, alpha)
    
    return alpha_new, accept_rate



def RobbinsMonroe(param, accept:torch.Tensor, n_step:int, acceptance_decreases_with_param:bool=True,
                  kappa:float=0.75, target_accept:float=0.234):
    
    param_log = torch.log(param)

    if acceptance_decreases_with_param: 
        param_new_log = param_log + (accept - torch.full_like(accept, target_accept)) / (n_step ** kappa)
    else:                               
        param_new_log = param_log - (accept - torch.full_like(accept, target_accept)) / (n_step ** kappa)

    return torch.exp(param_new_log)


def Adaptive_Burnin(BURNIN:int, K:int, SIGMA:int, TARGET_ACCEPT:float,
                    X:torch.Tensor, T:torch.Tensor, alpha_prior:tuple, CHAINS:int, CUDA:torch.device,
                    checkpoint:int, visualize:bool=True):
    
    P = X.shape[0]
    R = X.shape[1]
    C = X.shape[1]
    
    K_HISTORY = torch.zeros(CHAINS, BURNIN, 2)
    SIGMA_HISTORY = torch.zeros(CHAINS, BURNIN, 2)
    K_HISTORY[:, 0, 0] = K
    SIGMA_HISTORY[:, 0, 0] = SIGMA

    alpha = (torch.zeros(CHAINS, R, C) + torch.eye(R,C).unsqueeze(0)).to(CUDA)
    beta = (torch.ones(CHAINS, P, R, C) / C).to(CUDA)

    burnin_steps = tqdm.tqdm(range(1, BURNIN), desc="Burnin Steps",  position=0, leave=False)
    for step in burnin_steps:
        
        K = K_HISTORY[:, step - 1, 0].to(CUDA)
        SIGMA = SIGMA_HISTORY[:, step - 1, 0].to(CUDA)

        beta, acceptrate_beta = beta_step(beta, K, alpha, X, T)
        alpha, acceptrate_alpha = alpha_step(alpha, SIGMA, beta, alpha_prior)   

        K_HISTORY[:, step - 1, 1] = acceptrate_beta
        SIGMA_HISTORY[:, step - 1, 1] = acceptrate_alpha
        K_HISTORY[:, step, 0] = RobbinsMonroe(K.cpu(), acceptrate_beta, step, False, target_accept=TARGET_ACCEPT)
        SIGMA_HISTORY[:, step, 0] = RobbinsMonroe(SIGMA.cpu(), acceptrate_alpha, step, target_accept=TARGET_ACCEPT)

        if visualize: 
            if step % checkpoint == 0 or step == BURNIN - 1:
                render_burnin_diagnostic([K_HISTORY, SIGMA_HISTORY], step, TARGET_ACCEPT)
                display(burnin_steps.container)

    K = K_HISTORY[:, -1, 0].to(CUDA)
    SIGMA = SIGMA_HISTORY[:, -1, 0].to(CUDA)

    return K, SIGMA, alpha, beta



def GelmanRubin(samples:torch.Tensor):

    n_chains = samples.shape[0]
    n_steps = samples.shape[1]

    mean_chain = samples.mean(dim=1)

    mean_total = mean_chain.mean(dim=0)

    B = n_steps / (n_chains - 1) * ((mean_chain - mean_total.unsqueeze(0))**2).sum(dim=0)

    W = samples.var(dim=1, unbiased=True).mean(dim=0)

    R = ( ((n_steps - 1) / n_steps * W) + (B / n_steps) ) / W

    return torch.sqrt(R) 



def compute_histograms(ALPHAS:torch.Tensor, n_hist:int):

    CHAINS, STEP, R, C = ALPHAS.shape
    RC = R*C

    alpha = ALPHAS.clone().reshape(CHAINS, STEP, RC).permute(2, 0, 1)

    alpha_mins = alpha.reshape(RC, -1).min(dim=1).values 
    alpha_maxs = alpha.reshape(RC, -1).max(dim=1).values 

    hist_edges = torch.linspace(0, 1, n_hist+1)
    hist_edges = alpha_mins[:, None] + (alpha_maxs - alpha_mins)[:, None] * hist_edges
    hist_widths = hist_edges.diff()

    alpha_normalized = (alpha - alpha_mins[:, None, None]) / (alpha_maxs - alpha_mins)[:, None, None]
    alpha_histid = (alpha_normalized * n_hist).int().clamp(0, n_hist-1)
    alpha_hist = torch.zeros(RC, CHAINS, n_hist).scatter_add(2, alpha_histid, torch.ones(R*C, CHAINS, STEP))

    alpha_hist_density = alpha_hist / (STEP * hist_widths[:, None, :])
    alpha_hist_density_pooled = alpha_hist.sum(dim=1) / (CHAINS * STEP * hist_widths)

    histogram = torch.cat([alpha_hist_density, alpha_hist_density_pooled[:, None, :]], dim=1)

    hist_coords = (hist_edges[:, :-1] + hist_widths * 0.5).unsqueeze(1).expand_as(histogram)

    return histogram.permute(0, 2, 1), hist_coords.permute(0, 2, 1)


def render_diagnostics(diag, ALPHAS:torch.Tensor, 
                       step:int, CHAINS:int, R:int, C:int,
                       n_hist:int=50): #Visualization by Claude

    fig = plt.figure(figsize=(4 * C, 6 + 3*R), constrained_layout=True)
    fig.patch.set_facecolor("#0e1117")
    gs  = gridspec.GridSpec(2 + R, C, figure=fig)

    ax_acc_beta   = fig.add_subplot(gs[0, :C//2])
    ax_acc_alpha  = fig.add_subplot(gs[0, C//2:])
    ax_timing     = fig.add_subplot(gs[1, :C//2])
    ax_chain_info = fig.add_subplot(gs[1, C//2:])
    ax_hists      = [fig.add_subplot(gs[2 + n//C, n % C]) for n in range(R*C)]

    BG, TICK, SPINE = "#1a1d27", "#aab0c4", "#2e3250"

    def style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TICK)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE)

    for ax in [ax_acc_beta, ax_acc_alpha, ax_timing, ax_chain_info] + ax_hists:
        style(ax)

    # Acceptance Rate
    for ax, key, color, label in [
            (ax_acc_beta,  "accept_beta",  "#7eb8f7", "Beta Accept Rate"),
            (ax_acc_alpha, "accept_alpha", "#f7a97e", "Alpha Accept Rate"),
        ]:
        for chain in range(CHAINS):
            s_vals = [(s, v[chain]) for s, v in diag[key]]
            if s_vals:
                ss, vs = zip(*s_vals)
                ax.plot(ss, vs, alpha=0.4, linewidth=0.8, color=color)

        # Mean
        m_vals  = [(s, v.mean()) for s, v in diag[key]]
        ms, mv = zip(*m_vals)
        ax.plot(ms, mv, color="white", linewidth=1.5, label="mean")

        ax.axhline(0.234, color="#ff6b6b", linewidth=0.8,
                   linestyle="--", label="0.234 optimal")
        ax.set_ylim(0, 1)
        ax.set_title(label, color=color, fontsize=10)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor=BG, edgecolor=SPINE)


    # Time per Iteration
    if diag["step_times"]:
        ts, tv = zip(*diag["step_times"])
        ax_timing.plot(ts, tv, color="#a8d8a8", linewidth=1)

    ax_timing.set_title("Time per Step (s)", color="#a8d8a8", fontsize=10)


    # Last Acceptance
    last_beta  = [diag['accept_beta'][-1][1][chain] for chain in range(CHAINS)]
    last_alpha = [diag['accept_alpha'][-1][1][chain] for chain in range(CHAINS)]
    x = np.arange(CHAINS)

    ax_chain_info.bar(x - 0.2, last_beta,  0.4, color="#7eb8f7", label="beta",  alpha=0.8)
    ax_chain_info.bar(x + 0.2, last_alpha, 0.4, color="#f7a97e", label="alpha", alpha=0.8)

    ax_chain_info.set_xticks(x)
    ax_chain_info.set_xticklabels([f"C{i}" for i in x], fontsize=7, color=TICK)
    ax_chain_info.set_ylim(0, 1)
    ax_chain_info.set_title("Per-Chain Accept (last step)", color="#d4b8f7", fontsize=10)
    ax_chain_info.legend(fontsize=7, labelcolor="white", facecolor=BG, edgecolor=SPINE)


    # Histograms for Alpha
    GRStat = GelmanRubin(ALPHAS[:, :step+1, :, :])
    histogram_densities, histogram_coords = compute_histograms(ALPHAS[:, :step+1, :, :], n_hist) # Histogram Densities: RC x CHAINS+1 x n_hist

    for i, ax in enumerate(ax_hists):
        r, c = divmod(i, C)
    
        ax.plot(histogram_coords[i, :, :-1], histogram_densities[i, :, :-1], alpha=0.5, linewidth=0.7)
        ax.plot(histogram_coords[i, :, -1], histogram_densities[i, :, -1], color='white', linewidth=1.5)

        ax.set_title(f"α[{r},{c}]: {GRStat[r,c]}", color="#f7d97e", fontsize=9)
    
    fig.suptitle(f"MCMC Diagnostics — Step {step}", color="white",
                 fontsize=13, fontweight="bold")

    clear_output(wait=True)
    display(fig)
    plt.close(fig)   



def render_burnin_diagnostic(param_history:list, step:int, target_accept:float=0.234):

    n_param = len(param_history)

    fig = plt.figure(figsize=(16, 5*n_param), constrained_layout=True)
    fig.patch.set_facecolor("#0e1117")
    gs  = gridspec.GridSpec(n_param, 2, figure=fig)

    param_ax = [fig.add_subplot(gs[n, 0]) for n in range(n_param)]
    accept_ax = [fig.add_subplot(gs[n, 1]) for n in range(n_param)]

    for ax in param_ax + accept_ax:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(color="#aab0c4")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3250")

    for param, ax_value, ax_accept in zip(param_history, param_ax, accept_ax):

        for chain in range(param.shape[0]):
            ax_value.plot(range(step), param[chain, :step, 0].numpy(), color="#f7a97e", linewidth=0.8)
            ax_accept.plot(range(step), param[chain, :step, 1], color="#f7a97e", linewidth=0.8)

        mean_value = param[:, :step, 0].mean(dim=0).numpy()
        ax_value.plot(range(step), mean_value, color='white', linewidth=1.5, label="Mean")
        ax_value.set_title('Parameter', color='yellow', fontsize=10)
        ax_value.legend(fontsize=7, labelcolor="white",
                  facecolor="#1a1d27", edgecolor="#2e3250")

        mean_accept = param[:, :step, 1].mean(dim=0).numpy()
        ax_accept.plot(range(step), mean_accept, color='white', linewidth=1.5, label="Mean")
        ax_accept.axhline(target_accept, color='red', linestyle='--', label=f'Target Acceptance: {target_accept}')
        ax_accept.set_ylim(0, 1)
        ax_accept.set_title('Acceptance Rate', color='yellow', fontsize=10)
        ax_accept.legend(fontsize=7, labelcolor="white",
                  facecolor="#1a1d27", edgecolor="#2e3250")

    fig.suptitle(f"Adaptive Burnin — Step {step}", color="white",
                 fontsize=13, fontweight="bold")

    clear_output(wait=True)
    display(fig)
    plt.close(fig)



def EI_MCMC(X:torch.Tensor, T:torch.Tensor,
            CHAINS:int, STEPS:int, BURNIN:int, THINNING:int,
            alpha_prior:tuple,                       # Additions to improve mixing
            K:int, SIGMA:float, AdaptiveBurnin:bool=True, TARGET_ACCEPT:float=0.234,
            save_betas:bool=True, force_cpu:bool=False, visualize:bool=True, checkpoint:int=1000):
    

    CUDA = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P = X.shape[0]
    R = X.shape[1]
    C = T.shape[1]

    ALPHAS = torch.zeros(CHAINS, STEPS, R, C).cpu()
    if save_betas: BETAS = torch.zeros(CHAINS, STEPS, P, R, C).cpu()


    # Burnin
    if AdaptiveBurnin:
        K, SIGMA, alpha, beta = Adaptive_Burnin(BURNIN, K, SIGMA, TARGET_ACCEPT, X, T, alpha_prior, CHAINS, CUDA, checkpoint, visualize)

    else:
        K = torch.full((CHAINS,), K).to(CUDA)
        SIGMA = torch.full((CHAINS,), SIGMA).to(CUDA)

        alpha = (torch.zeros(CHAINS, R, C) + torch.eye(R,C).unsqueeze(0)).to(CUDA)
        beta = (torch.ones(CHAINS, P, R, C) / C).to(CUDA)

        burnin_steps = tqdm.tqdm(range(1, BURNIN), desc="Burnin Steps",  position=0, leave=False)
        for step in burnin_steps:
            beta, _ = beta_step(beta, K, alpha, X, T)
            alpha, _ = alpha_step(alpha, SIGMA, beta, alpha_prior) 
              

    ALPHAS[:, 0, :, :] = alpha.clone().cpu()
    if save_betas: BETAS[:, 0, :, :, :] = beta.clone().cpu()



    # Main Loop
    if visualize:
        diag = {
            "accept_beta":  [],   # (step, list of values)
            "accept_alpha": [],
            "step_times":   [],   # (step, seconds)
        }

        progress_steps  = tqdm.tqdm(range(1, STEPS), desc="Steps",  position=0)
        for step in progress_steps:
            t_start = time.perf_counter()

            beta_new,  acceptrate_beta  = beta_step (beta,  K, alpha, X, T)
            alpha_new, acceptrate_alpha = alpha_step(alpha, SIGMA, beta_new, alpha_prior)

            beta  = beta_new
            alpha = alpha_new

            if save_betas: BETAS [:, step, :, :, :] = beta_new.clone().cpu()
            ALPHAS[:, step, :, :] = alpha_new.clone().cpu()


            diag["accept_beta" ].append((step, acceptrate_beta))
            diag["accept_alpha"].append((step, acceptrate_alpha))
            t_elapsed = time.perf_counter() - t_start
            diag["step_times"].append((step, t_elapsed))


            if step == STEPS - 1 or step % checkpoint == 0:

                step_beta_vals  = diag["accept_beta"][-1][1].mean().numpy()
                step_alpha_vals = diag["accept_alpha"][-1][1].mean().numpy()
                progress_steps.set_postfix({
                    "step_time": f"{t_elapsed:.3f}s",
                    "β_accept":  f"{step_beta_vals:.3f}"  if step_beta_vals  else "—",
                    "α_accept":  f"{step_alpha_vals:.3f}" if step_alpha_vals else "—",
                })

                render_diagnostics(diag, ALPHAS, step, CHAINS, R, C)
                display(progress_steps.container)


    else:
        progress_steps  = tqdm.tqdm(range(1, STEPS), desc="Steps",  position=0)
        for step in progress_steps:

            beta_new,  acceptrate_beta  = beta_step (beta,  K, alpha, X, T)
            alpha_new, acceptrate_alpha = alpha_step(alpha, SIGMA, beta_new, alpha_prior)

            beta  = beta_new
            alpha = alpha_new

            if save_betas: BETAS [:, step, :, :, :] = beta_new.clone().cpu()
            ALPHAS[:, step, :, :] = alpha_new.clone().cpu()


    if save_betas: return ALPHAS, BETAS
    else: return ALPHAS