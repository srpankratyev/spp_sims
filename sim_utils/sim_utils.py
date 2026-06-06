# sim_utils.py — Simulation code for switchback power formula paper
# Loaded via %run ./sim_utils/sim_utils in the notebook
SIM_UTILS_VERSION = "2.1"

import os
import time
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict, replace
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats as sp_stats

# =============================================================================
# Regime Parameters
# =============================================================================

@dataclass
class RegimeParams:
    n_sp: int = 200
    n_hours: int = 24
    mean_obs_per_sp_hour: int = 180
    cluster_size_cv: float = 1.5

    grand_mean: float = 2000.0
    total_std: float = 1000.0

    var_share_sp: float = 0.05
    var_share_hour: float = 0.03
    var_share_time_shock: float = 0.0   # random hour shocks shared across SPs
    var_share_interaction: float = 0.20
    var_share_residual: float = 0.72

    acf_lag1: float = 0.3

    tau: float = 0.0
    tau_sd: float = 0.0
    treatment_prob: float = 0.5
    block_length: int = 1               # 1 = cell-level; >1 = treatment constant for L hours

    error_df: float = 0.0               # 0 = Gaussian; >0 = t-distribution with this df

    @property
    def sigma_sp(self):
        return self.total_std * np.sqrt(self.var_share_sp)

    @property
    def sigma_hour(self):
        return self.total_std * np.sqrt(self.var_share_hour)

    @property
    def sigma_time_shock(self):
        return self.total_std * np.sqrt(self.var_share_time_shock)

    @property
    def sigma_interaction(self):
        return self.total_std * np.sqrt(self.var_share_interaction)

    @property
    def sigma_residual(self):
        return self.total_std * np.sqrt(self.var_share_residual)


# =============================================================================
# Data Generating Process
# =============================================================================

def generate_experiment(params: RegimeParams, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    J = params.n_sp
    H = params.n_hours

    cv = params.cluster_size_cv
    sigma_ln = np.sqrt(np.log(1 + cv**2))
    mu_ln = np.log(params.mean_obs_per_sp_hour) - sigma_ln**2 / 2
    sp_mean_sizes = rng.lognormal(mu_ln, sigma_ln, size=J)
    sp_mean_sizes = np.maximum(sp_mean_sizes, 1).astype(int)

    lam_matrix = sp_mean_sizes[:, None] * np.ones((1, H))
    cell_sizes = np.maximum(rng.poisson(lam=lam_matrix), 1)

    alpha = rng.normal(0, params.sigma_sp, size=J)

    gamma = rng.normal(0, params.sigma_hour, size=H)

    rho = params.acf_lag1
    innovation_std = params.sigma_interaction * np.sqrt(1 - rho**2)
    delta = np.zeros((J, H))
    delta[:, 0] = rng.normal(0, params.sigma_interaction, size=J)
    for h in range(1, H):
        delta[:, h] = rho * delta[:, h-1] + rng.normal(0, innovation_std, size=J)

    L = params.block_length
    if L > 1:
        n_blocks = int(np.ceil(H / L))
        block_assignments = rng.binomial(1, params.treatment_prob, size=(J, n_blocks))
        treatment_cell = np.repeat(block_assignments, L, axis=1)[:, :H]
    else:
        treatment_cell = rng.binomial(1, params.treatment_prob, size=(J, H))

    flat_sizes = cell_sizes.ravel()
    total_obs = flat_sizes.sum()

    cell_sp = np.repeat(np.arange(J), H)
    cell_hour = np.tile(np.arange(H), J)

    sp_ids = np.repeat(cell_sp, flat_sizes)
    hour_ids = np.repeat(cell_hour, flat_sizes)
    treatments = np.repeat(treatment_cell.ravel(), flat_sizes)
    alpha_arr = np.repeat(alpha[cell_sp], flat_sizes)
    gamma_arr = np.repeat(gamma[cell_hour], flat_sizes)
    delta_arr = np.repeat(delta.ravel(), flat_sizes)

    if params.error_df > 0:
        raw_t = rng.standard_t(params.error_df, size=total_obs)
        epsilon = raw_t * (params.sigma_residual / np.sqrt(params.error_df / (params.error_df - 2)))
    else:
        epsilon = rng.normal(0, params.sigma_residual, size=total_obs)

    if params.tau_sd > 0:
        tau_j = params.tau + params.tau_sd * rng.normal(0, 1, size=J)
    else:
        tau_j = np.full(J, params.tau)
    tau_arr = np.repeat(tau_j[cell_sp], flat_sizes)

    y_obs = (params.grand_mean + alpha_arr + gamma_arr + delta_arr
             + tau_arr * treatments + epsilon)

    return pd.DataFrame({
        'sp_id': sp_ids, 'hour': hour_ids, 'treatment': treatments,
        'y_obs': y_obs,
    })


# =============================================================================
# Estimators
# =============================================================================

@dataclass
class EstimateResult:
    method: str
    ate_hat: float
    se: float
    ci_lo: float
    ci_hi: float
    pval: float
    rejected: bool


def _ols_cluster_robust(y, treatment, clusters):
    X = add_constant(treatment)
    results = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': clusters})
    ate = results.params[1]
    se = results.bse[1]
    ci = results.conf_int(alpha=0.05)
    return EstimateResult(
        method='', ate_hat=ate, se=se,
        ci_lo=ci[1, 0], ci_hi=ci[1, 1],
        pval=results.pvalues[1], rejected=results.pvalues[1] < 0.05
    )


def estimate_raw(df):
    r = _ols_cluster_robust(df['y_obs'].values, df['treatment'].values, df['sp_id'].values)
    r.method = 'raw'
    return r


ALL_ESTIMATORS = [estimate_raw]
ALL_METHODS = ['raw']
METHOD_COLORS = {'raw': '#9E9E9E'}
METHOD_LABELS = {'raw': 'Raw'}


# =============================================================================
# Simulation Runner + Checkpointing
# =============================================================================

SIM_RESULTS_DIR = './output/sim_results/'


def _checkpoint_path(regime_key, method, results_dir=None):
    d = results_dir if results_dir is not None else SIM_RESULTS_DIR
    return os.path.join(d, f'{regime_key}_{method}.pkl')


def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def _save_checkpoint(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def run_mc_for_method(regime_key, params, estimator_fn, n_sims=200,
                      regime_index=0, save_every=5, print_every=20,
                      results_dir=None, force_restart=False):
    rd = results_dir if results_dir is not None else SIM_RESULTS_DIR
    os.makedirs(rd, exist_ok=True)
    method_name = estimator_fn.__name__.replace('estimate_', '')
    ckpt_path = _checkpoint_path(regime_key, method_name, rd)

    if force_restart and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f'[{regime_key}/{method_name}] Force restart — deleted existing checkpoint.')

    ckpt = _load_checkpoint(ckpt_path)
    if ckpt is not None:
        records = ckpt['results'].to_dict('records')
        done = len(records)
        if done >= n_sims:
            print(f'[{regime_key}/{method_name}] Already complete ({done}/{n_sims}). Skipping.')
            return ckpt['results']
        print(f'[{regime_key}/{method_name}] Resuming from sim {done}/{n_sims}')
    else:
        records = []
        done = 0
        print(f'[{regime_key}/{method_name}] Starting {n_sims} sims')

    t0 = time.time()
    for i in range(done, n_sims):
        seed = regime_index * n_sims + i
        t_sim = time.time()
        df = generate_experiment(params, seed=seed)
        t_gen = time.time() - t_sim
        r = estimator_fn(df)
        t_est = time.time() - t_sim - t_gen

        records.append({
            'sim_id': i, 'seed': seed,
            'method': r.method, 'ate_hat': r.ate_hat,
            'se': r.se, 'ci_lo': r.ci_lo, 'ci_hi': r.ci_hi,
            'pval': r.pval, 'rejected': r.rejected,
            't_gen': round(t_gen, 4), 't_est': round(t_est, 4),
        })

        if (i + 1) % save_every == 0 or (i + 1) == n_sims:
            _save_checkpoint(ckpt_path, {
                'regime_key': regime_key, 'params': asdict(params),
                'method': method_name, 'n_sims_target': n_sims,
                'results': pd.DataFrame(records),
            })

        if (i + 1) % print_every == 0 or (i + 1) == n_sims:
            elapsed = time.time() - t0
            rate = (i + 1 - done) / elapsed if elapsed > 0 else 0
            eta = (n_sims - i - 1) / rate if rate > 0 else 0
            print(f'  [{regime_key}/{method_name}] {i+1}/{n_sims}'
                  f'  ({rate:.1f} sims/sec, ETA {eta:.0f}s)'
                  f'  [last: gen={t_gen:.1f}s est={t_est:.1f}s]')

    elapsed = time.time() - t0
    new_sims = n_sims - done
    if elapsed > 0 and new_sims > 0:
        print(f'  Done: {new_sims} sims in {elapsed:.1f}s ({new_sims/elapsed:.1f} sims/sec)')
    return pd.DataFrame(records)


def load_regime_results(regime_key, methods=None, results_dir=None):
    if methods is None:
        methods = ALL_METHODS
    dfs = []
    for method in methods:
        ckpt = _load_checkpoint(_checkpoint_path(regime_key, method, results_dir))
        if ckpt is not None:
            dfs.append(ckpt['results'])
        else:
            print(f'  Warning: no results for {regime_key}/{method}')
    if not dfs:
        raise FileNotFoundError(f'No results found for regime {regime_key}')
    return pd.concat(dfs, ignore_index=True)


def summarize_regime(regime_key, tau, methods=None, results_dir=None):
    mc = load_regime_results(regime_key, methods, results_dir=results_dir)
    available = [m for m in (methods or ALL_METHODS) if m in mc['method'].unique()]
    se_raw = mc[mc['method'] == 'raw']['se'].mean() if 'raw' in mc['method'].values else None

    print(f"\n{'=' * 70}")
    print(f'REGIME: {regime_key}  (tau = {tau})')
    print(f"{'=' * 70}")

    for method in available:
        sub = mc[mc['method'] == method]
        n = len(sub)
        mean_ate, std_ate, mean_se = sub['ate_hat'].mean(), sub['ate_hat'].std(), sub['se'].mean()
        reject_rate = sub['rejected'].mean()
        covers = ((sub['ci_lo'] <= tau) & (tau <= sub['ci_hi'])).mean()

        label = 'FPR' if tau == 0 else 'Power'
        se_ratio = f'{mean_se / se_raw:.3f}' if se_raw else 'N/A'

        print(f"\n  [{METHOD_LABELS.get(method, method):>6s}] (n={n})")
        print(f'    Bias:      {mean_ate - tau:+.3f}')
        print(f'    Mean SE:   {mean_se:.3f}   SE/SE_raw: {se_ratio}')
        print(f'    Emp. SE:   {std_ate:.3f}')
        print(f'    {label}:{"":>6s} {reject_rate:.3f}')
        print(f'    Coverage:  {covers:.3f}')
        print(f'    MDE(80%):  {2.80 * mean_se:.2f} ({2.80 * mean_se / 2000 * 100:.2f}% of grand mean)')


print('sim_utils loaded.')
