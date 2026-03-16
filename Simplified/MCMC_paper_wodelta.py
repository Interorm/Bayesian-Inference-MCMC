import torch
from torch.special import gammaln
import torch.distributions as dist

import tqdm.auto as tqdm 
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, clear_output
import numpy as np


def compute_eta(alpha:torch.Tensor, density:torch.Tensor):

    eta_free = density[:, :, None] * alpha.exp()
    eta_ref = density[:, :, None].expand(density.shape[0], density.shape[1], 1)

    return torch.cat([eta_free, eta_ref], dim=2)



def propose_beta(beta:torch.Tensor, k:torch.Tensor):

    CHAINS, P, R, C = beta.shape
    concentration = (beta * k[:, None, None, None]).reshape(CHAINS, P * R, C)

    beta_prop = dist.Dirichlet( concentration ).sample()

    return beta_prop.reshape(CHAINS, P, R, C)


def acceptance_beta(beta:torch.Tensor, beta_prop:torch.Tensor,
                    alpha:torch.Tensor, density:torch.Tensor, 
                    X:torch.Tensor, T:torch.Tensor, 
                    k:torch.Tensor):
    
    k = k.clone()[:, None, None, None]

    beta_log = torch.log(beta)
    beta_prop_log = torch.log(beta_prop)

    eta = compute_eta(alpha, density)
    log_prior = (eta - 1).unsqueeze(1) * (beta_prop_log - beta_log)

    theta = (X[None, :, :, None] * beta).sum(dim=2)
    theta_prop = theta[:, :, None, :] + (X[None, :, :, None] * (beta_prop - beta))
    log_likelihood = T[None, :, None, :] * (torch.log(theta_prop) - torch.log(theta[: ,:, None, :]))

    log_proposal = k * (beta_prop * beta_log - beta * beta_prop_log) + \
                   gammaln(k * beta) - gammaln(k * beta_prop)
    

    log_acceptance = (log_prior + log_likelihood + log_proposal).sum(dim=3)

    return log_acceptance


def beta_step(beta:torch.Tensor, k:torch.Tensor,
              alpha:torch.Tensor, density:torch.Tensor, X:torch.Tensor, T:torch.Tensor):
    
    beta_prop = propose_beta(beta, k)

    acceptance = acceptance_beta(beta, beta_prop, alpha, density, X, T, k)

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


def acceptance_alpha(alpha:torch.Tensor, alpha_prop:torch.Tensor, 
                     beta:torch.Tensor, density:torch.Tensor, 
                     alpha_prior:tuple):

    CHAINS, P, R, C = beta.shape
    alpha_prior_mu, alpha_prior_sigma = alpha_prior

    eta = compute_eta(alpha, density)        
    eta_prop = compute_eta(alpha_prop, density)

    eta_expandedRCC = eta.unsqueeze(2).expand(CHAINS, R, C, C).clone()
    torch.diagonal(eta_expandedRCC, dim1=2, dim2=3).copy_(eta_prop)


    log_joined_gamma = P * (gammaln(eta_expandedRCC.sum(dim=3)) - gammaln(eta.sum(dim=2))[:, :, None])

    log_indiv_gamma = P * (gammaln(eta) - gammaln(eta_prop))
    
    log_kernel = (eta_prop - eta) * torch.log(beta).sum(dim=1)


    log_prior = 0.5 / alpha_prior_sigma**2 * ( (alpha - alpha_prior_mu)**2 - (alpha_prop - alpha_prior_mu)**2 )


    log_acceptance = (log_joined_gamma + log_indiv_gamma + log_kernel)[:, :, :-1] + log_prior

    return log_acceptance


def alpha_step(alpha:torch.Tensor, sigma:torch.Tensor, 
               beta:torch.Tensor, density:torch.Tensor, 
               alpha_prior:tuple):

    alpha_prop = propose_alpha(alpha, sigma)

    acceptance = acceptance_alpha(alpha, alpha_prop, beta, density, alpha_prior)

    acceptance_threshhold = torch.log(dist.Uniform(torch.zeros_like(acceptance),
                                                   torch.ones_like(acceptance)).sample())
    acceptance_mask = acceptance_threshhold <= acceptance 
    accept_rate = acceptance_mask.float().mean(dim=(1,2)).cpu()
    alpha_new = torch.where(acceptance_mask, alpha_prop, alpha)
    
    return alpha_new, accept_rate



def propose_density(density:torch.Tensor, sigma:torch.Tensor):

    density_prop = dist.Normal(
                            loc = density,
                            scale = sigma[:, None].expand_as(density)
                        ).sample().clamp(min=1e-6)

    return density_prop


def acceptance_density(density:torch.Tensor, density_prop:torch.Tensor,
                       alpha:torch.Tensor, beta:torch.Tensor, 
                       density_lambda:float):
    
    CHAINS, P, R, C = beta.shape

    eta = compute_eta(alpha, density)
    eta_prop = compute_eta(alpha, density_prop)


    log_gammasum = P * (gammaln(eta_prop.sum(dim=2)) - gammaln(eta.sum(dim=2)))

    log_sumgamma = P * (gammaln(eta) - gammaln(eta_prop)).sum(dim=2)

    log_kernel = ( (eta_prop - eta) * beta.log().sum(dim=1) ).sum(dim=2)


    log_prior = (density - density_prop) * density_lambda


    log_acceptance = log_gammasum + log_sumgamma + log_kernel + log_prior

    return log_acceptance


def density_step(density:torch.Tensor, sigma:torch.Tensor, 
                 alpha:torch.Tensor, beta:torch.Tensor, 
                 density_lambda:float):

    density_prop = propose_density(density, sigma)

    acceptance = acceptance_density(density, density_prop, alpha, beta, density_lambda)

    acceptance_threshhold = torch.log(dist.Uniform(torch.zeros_like(acceptance),
                                                   torch.ones_like(acceptance)).sample())
    acceptance_mask = acceptance_threshhold <= acceptance 
    accept_rate = acceptance_mask.float().mean(dim=1).cpu()
    density_new = torch.where(acceptance_mask, density_prop, density)
    
    return density_new, accept_rate



def RobbinsMonroe(param, accept:torch.Tensor, n_step:int, acceptance_decreases_with_param:bool=True,
                  kappa:float=0.75, target_accept:float=0.234):
    
    param_log = torch.log(param)

    if acceptance_decreases_with_param: 
        param_new_log = param_log + (accept - torch.full_like(accept, target_accept)) / (n_step ** kappa)
    else:                               
        param_new_log = param_log - (accept - torch.full_like(accept, target_accept)) / (n_step ** kappa)

    return torch.exp(param_new_log)


def Adaptive_Burnin(BURNIN:int, K:int, SIGMA_ALPHA:float, SIGMA_DENSITY:float, TARGET_ACCEPT:float,
                    X:torch.Tensor, T:torch.Tensor, alpha_prior:tuple, density_lambda:float, 
                    CHAINS:int, CUDA:torch.device,
                    checkpoint:int, visualize:bool=True):
    
    P = X.shape[0]
    R = X.shape[1]
    C = T.shape[1]
    

    K_HISTORY = torch.zeros(CHAINS, BURNIN, 2)
    SIGMA_ALPHA_HISTORY = torch.zeros(CHAINS, BURNIN, 2)
    SIGMA_DENSITY_HISTORY = torch.zeros(CHAINS, BURNIN, 2)

    K_HISTORY[:, 0, 0] = K
    SIGMA_ALPHA_HISTORY[:, 0, 0] = SIGMA_ALPHA
    SIGMA_DENSITY_HISTORY[:, 0, 0] = SIGMA_DENSITY


    alpha = (torch.zeros(CHAINS, R, C - 1) + torch.eye(R,C - 1).unsqueeze(0)).to(CUDA)
    beta = (torch.ones(CHAINS, P, R, C) / C).to(CUDA)
    density = torch.ones(CHAINS, R).to(CUDA)

    burnin_steps = tqdm.tqdm(range(1, BURNIN), desc="Burnin Steps",  position=0, leave=False)
    for step in burnin_steps:
        
        K = K_HISTORY[:, step - 1, 0].to(CUDA)
        SIGMA_ALPHA = SIGMA_ALPHA_HISTORY[:, step - 1, 0].to(CUDA)
        SIGMA_DENSITY = SIGMA_DENSITY_HISTORY[:, step - 1, 0].to(CUDA)

        beta, acceptrate_beta = beta_step(beta, K, alpha, density, X, T)
        alpha, acceptrate_alpha = alpha_step(alpha, SIGMA_ALPHA, beta, density, alpha_prior)   
        density, acceptrate_density = density_step(density, SIGMA_DENSITY, alpha, beta, density_lambda)

        K_HISTORY[:, step - 1, 1] = acceptrate_beta
        SIGMA_ALPHA_HISTORY[:, step - 1, 1] = acceptrate_alpha
        SIGMA_DENSITY_HISTORY[:, step - 1, 1] = acceptrate_density

        K_HISTORY[:, step, 0] = RobbinsMonroe(K.cpu(), acceptrate_beta, step, False, target_accept=TARGET_ACCEPT)
        SIGMA_ALPHA_HISTORY[:, step, 0] = RobbinsMonroe(SIGMA_ALPHA.cpu(), acceptrate_alpha, step, target_accept=TARGET_ACCEPT)
        SIGMA_DENSITY_HISTORY[:, step, 0] = RobbinsMonroe(SIGMA_DENSITY.cpu(), acceptrate_density, step, target_accept=TARGET_ACCEPT)

        if visualize: 
            if step % checkpoint == 0 or step == BURNIN - 1:
                render_burnin_diagnostic([K_HISTORY, SIGMA_ALPHA_HISTORY, SIGMA_DENSITY_HISTORY], step, TARGET_ACCEPT)
                display(burnin_steps.container)

    K = K_HISTORY[:, -1, 0].to(CUDA)
    SIGMA_ALPHA = SIGMA_ALPHA_HISTORY[:, -1, 0].to(CUDA)
    SIGMA_DENSITY = SIGMA_DENSITY_HISTORY[:, -1, 0].to(CUDA)

    return K, SIGMA_ALPHA, SIGMA_DENSITY, alpha, beta, density



def GelmanRubin(samples:torch.Tensor):

    n_chains = samples.shape[0]
    n_steps = samples.shape[1]

    mean_chain = samples.mean(dim=1)

    mean_total = mean_chain.mean(dim=0)

    B = n_steps / (n_chains - 1) * ((mean_chain - mean_total.unsqueeze(0))**2).sum(dim=0)

    W = samples.var(dim=1, unbiased=True).mean(dim=0)

    R = ( ((n_steps - 1) / n_steps * W) + (B / n_steps) ) / W

    return torch.sqrt(R) 


def compute_histograms(PARAM:torch.Tensor, n_hist:int):

    # Checks for Size: ALPHAS has 4, DENSITY has 3 dimensions
    if PARAM.dim() == 4:
        CHAINS, STEP, R, C = PARAM.shape
        RC = R*C

        param = PARAM.clone().reshape(CHAINS, STEP, RC).permute(2, 0, 1)
    else:
        CHAINS, STEP, R = PARAM.shape
        RC = R

        param = PARAM.clone().permute(2, 0, 1)
        
    alpha_mins = param.reshape(RC, -1).min(dim=1).values                # Legacy Naming; Change in final
    alpha_maxs = param.reshape(RC, -1).max(dim=1).values 

    hist_edges = torch.linspace(0, 1, n_hist+1)
    hist_edges = alpha_mins[:, None] + (alpha_maxs - alpha_mins)[:, None] * hist_edges
    hist_widths = hist_edges.diff()

    alpha_normalized = (param - alpha_mins[:, None, None]) / (alpha_maxs - alpha_mins)[:, None, None]
    alpha_histid = (alpha_normalized * n_hist).int().clamp(0, n_hist-1)
    alpha_hist = torch.zeros(RC, CHAINS, n_hist).scatter_add(2, alpha_histid, torch.ones(RC, CHAINS, STEP))

    alpha_hist_density = alpha_hist / (STEP * hist_widths[:, None, :])
    alpha_hist_density_pooled = alpha_hist.sum(dim=1) / (CHAINS * STEP * hist_widths)

    histogram = torch.cat([alpha_hist_density, alpha_hist_density_pooled[:, None, :]], dim=1)

    hist_coords = (hist_edges[:, :-1] + hist_widths * 0.5).unsqueeze(1).expand_as(histogram)

    return histogram.permute(0, 2, 1), hist_coords.permute(0, 2, 1)


def render_diagnostics(diag, ALPHAS:torch.Tensor, DENSITY:torch.Tensor, step:int, 
                       n_hist:int=50): #Visualization by Claude
    
    CHAINS, _, R, C = ALPHAS.shape

    fig = plt.figure(figsize=(4 * C + 4, 9 + 3*R), constrained_layout=True)
    fig.patch.set_facecolor("#0e1117")
    gs  = gridspec.GridSpec(3 + R, C + 1, figure=fig)

    ax_timing     = fig.add_subplot(gs[0, :C//2 + 1])
    ax_chain_info = fig.add_subplot(gs[0, C//2 + 1:])   

    ax_acc_beta   = fig.add_subplot(gs[1, :])
    ax_acc_alpha  = fig.add_subplot(gs[2, :C//2 + 1])
    ax_acc_density = fig.add_subplot(gs[2, C//2 + 1:])

    ax_hists_density = [fig.add_subplot(gs[3 + r, C]) for r in range(R)]
    ax_hists_alpha = [fig.add_subplot(gs[3 + r, c]) for r in range(R) for c in range(C)]

    BG, TICK, SPINE = "#1a1d27", "#aab0c4", "#2e3250"

    def style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TICK)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE)

    for ax in [ax_acc_beta, ax_acc_alpha, ax_acc_density, ax_timing, ax_chain_info] + ax_hists_alpha + ax_hists_density:
        style(ax)

    # Acceptance Rate
    for ax, key, color, label in [
            (ax_acc_beta,  "accept_beta",  "#7eb8f7", "Beta Accept Rate"),
            (ax_acc_alpha, "accept_alpha", "#f7a97e", "Alpha Accept Rate"),
            (ax_acc_density, "accept_density", "#cc8cb6", "Density Accept Rate"),
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
    last_density = [diag['accept_density'][-1][1][chain] for chain in range(CHAINS)]
    x = np.arange(CHAINS)

    ax_chain_info.bar(x - 0.3, last_beta,  0.3, color="#7eb8f7", label="beta",  alpha=0.8)
    ax_chain_info.bar(x , last_alpha,  0.3, color="#f7a97e", label="alpha",  alpha=0.8)
    ax_chain_info.bar(x + 0.3, last_density, 0.3, color="#cc8cb6", label="density", alpha=0.8)

    ax_chain_info.set_xticks(x)
    ax_chain_info.set_xticklabels([f"C{i}" for i in x], fontsize=7, color=TICK)
    ax_chain_info.set_ylim(0, 1)
    ax_chain_info.set_title("Per-Chain Accept (last step)", color="#d4b8f7", fontsize=10)
    ax_chain_info.legend(fontsize=7, labelcolor="white", facecolor=BG, edgecolor=SPINE)


    # Histograms for Alpha
    GRStat_Alpha = GelmanRubin(ALPHAS[:, :step+1, :, :])
    alpha_densities, alpha_hist_coords = compute_histograms(ALPHAS[:, :step+1, :, :], n_hist) # Histogram Densities: RC x CHAINS+1 x n_hist

    for i, ax in enumerate(ax_hists_alpha):
        r, c = divmod(i, C)
    
        ax.plot(alpha_hist_coords[i, :, :-1], alpha_densities[i, :, :-1], alpha=0.5, linewidth=0.7)
        ax.plot(alpha_hist_coords[i, :, -1], alpha_densities[i, :, -1], color='white', linewidth=1.5)

        ax.set_title(f"α[{r},{c}]: {GRStat_Alpha[r,c]}", color="#f7d97e", fontsize=9)
    

    # Histograms for Density
    GRStat_Density = GelmanRubin(DENSITY[:, :step+1, :].unsqueeze(-1)).squeeze(-1)
    density_densities, density_hist_coords = compute_histograms(DENSITY[:, :step + 1, :], n_hist)
    
    for r, ax in enumerate(ax_hists_density):
        ax.plot(density_hist_coords[r, :, :-1], density_densities[r, :, :-1], alpha=0.5, linewidth=0.7)
        ax.plot(density_hist_coords[r, :, -1], density_densities[r, :, -1], color='white', linewidth=1.5)

        ax.set_title(f"d[{r}]: {GRStat_Density[r]:.3f}", color="#a8f7d9", fontsize=9)


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
            alpha_prior:tuple, density_lambda,                      # Additions to improve mixing
            K:float, SIGMA_ALPHA:float, SIGMA_DENSITY:float, AdaptiveBurnin:bool=True, TARGET_ACCEPT:float=0.234,
            save_betas:bool=True, force_cpu:bool=False, visualize:bool=True, checkpoint:int=1000):
    

    CUDA = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P = X.shape[0]
    R = X.shape[1]
    C = T.shape[1]

    ALPHAS = torch.zeros(CHAINS, STEPS, R, C - 1).cpu()
    DENSITY = torch.zeros(CHAINS, STEPS, R).cpu()
    if save_betas: BETAS = torch.zeros(CHAINS, STEPS, P, R, C).cpu()


    # Burnin
    if AdaptiveBurnin:
        K, SIGMA_ALPHA, SIGMA_DENSITY, alpha, beta, density = Adaptive_Burnin(BURNIN, K, SIGMA_ALPHA, SIGMA_DENSITY, TARGET_ACCEPT, 
                                                                              X, T, alpha_prior, density_lambda, 
                                                                              CHAINS, CUDA, checkpoint, visualize)

    else:
        K = torch.full((CHAINS,), K).to(CUDA)
        SIGMA_ALPHA = torch.full((CHAINS,), SIGMA_ALPHA).to(CUDA)
        SIGMA_DENSITY = torch.full((CHAINS,), SIGMA_DENSITY).to(CUDA)

        alpha = (torch.zeros(CHAINS, R, C - 1) + torch.eye(R,C - 1).unsqueeze(0)).to(CUDA)
        beta = (torch.ones(CHAINS, P, R, C) / C).to(CUDA)
        density = torch.ones(CHAINS, R).to(CUDA)

        burnin_steps = tqdm.tqdm(range(1, BURNIN), desc="Burnin Steps",  position=0, leave=False)
        for step in burnin_steps:
            beta, _ = beta_step(beta, K, alpha, density, X, T)
            alpha, _ = alpha_step(alpha, SIGMA_ALPHA, beta, density, alpha_prior) 
            density, _ = density_step(density, SIGMA_DENSITY, alpha, beta, density_lambda)
              

    ALPHAS[:, 0, :, :] = alpha.clone().cpu()
    DENSITY[:, 0, :] = density.clone().cpu()
    if save_betas: BETAS[:, 0, :, :, :] = beta.clone().cpu()



    # Main Loop
    if visualize:
        diag = {
            "accept_beta":  [],   # (step, list of values)
            "accept_alpha": [],
            "accept_density": [],
            "step_times":   [],   # (step, seconds)
        }

        progress_steps  = tqdm.tqdm(range(1, STEPS), desc="Steps",  position=0)
        for step in progress_steps:
            t_start = time.perf_counter()

            beta,  acceptrate_beta  = beta_step (beta,  K, alpha, density, X, T)
            alpha, acceptrate_alpha = alpha_step(alpha, SIGMA_ALPHA, beta, density, alpha_prior)
            density, acceptrate_density = density_step(density, SIGMA_DENSITY, alpha, beta, density_lambda)

            if save_betas: BETAS [:, step, :, :, :] = beta.clone().cpu()
            ALPHAS[:, step, :, :] = alpha.clone().cpu()
            DENSITY[:, step, :] = density.clone().cpu()


            diag["accept_beta" ].append((step, acceptrate_beta))
            diag["accept_alpha"].append((step, acceptrate_alpha))
            diag["accept_density"].append((step, acceptrate_density))
            t_elapsed = time.perf_counter() - t_start
            diag["step_times"].append((step, t_elapsed))


            if step == STEPS - 1 or step % checkpoint == 0:

                step_beta_vals  = diag["accept_beta"][-1][1].mean().numpy()
                step_alpha_vals = diag["accept_alpha"][-1][1].mean().numpy()
                step_density_vals = diag["accept_density"][-1][1].mean().numpy()
                progress_steps.set_postfix({
                    "step_time": f"{t_elapsed:.3f}s",
                    "β_accept":  f"{step_beta_vals:.3f}"  if step_beta_vals  else "—",
                    "α_accept":  f"{step_alpha_vals:.3f}" if step_alpha_vals else "—",
                    "d_accept": f"{step_density_vals:.3f}" if step_alpha_vals else "—"
                })

                render_diagnostics(diag, ALPHAS, DENSITY, step)
                display(progress_steps.container)


    else:
        progress_steps  = tqdm.tqdm(range(1, STEPS), desc="Steps",  position=0)
        for step in progress_steps:

            beta,  acceptrate_beta  = beta_step (beta,  K, alpha, density, X, T)
            alpha, acceptrate_alpha = alpha_step(alpha, SIGMA_ALPHA, beta, density, alpha_prior)
            density, acceptrate_density = density_step(density, SIGMA_DENSITY, alpha, beta, density_lambda)


            if save_betas: BETAS [:, step, :, :, :] = beta.clone().cpu()
            ALPHAS[:, step, :, :] = alpha.clone().cpu()
            DENSITY[:, step, :] = density.clone().cpu()


    if save_betas: return ALPHAS, DENSITY, BETAS
    else: return ALPHAS, DENSITY