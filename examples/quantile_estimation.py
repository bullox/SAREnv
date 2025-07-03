import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sarenv import (
    CLIMATE_DRY,
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
    DataGenerator,
    get_logger,
)
from sarenv.utils.lost_person_behavior import get_environment_radius
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, gamma
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm, gamma
from scipy.optimize import minimize

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def plot_fit_errors_pretty():
    plt.style.use('seaborn-whitegrid')  # clean style with grid
    environments = [ENVIRONMENT_TYPE_FLAT, ENVIRONMENT_TYPE_MOUNTAINOUS]
    climates = [CLIMATE_DRY, CLIMATE_TEMPERATE]
    percentiles = np.array([0.25, 0.5, 0.75, 0.95])

    errors = {
        "Normal": [],
        "Log-normal": [],
        "Gamma": [],
    }
    labels = []

    def normal_fit(climate, environment):
        values = get_environment_radius(environment, climate)
        z = norm.ppf(percentiles)
        A = np.vstack([np.ones_like(z), z]).T
        mu, sigma = np.linalg.lstsq(A, values, rcond=None)[0]
        fitted = mu + sigma * z
        return calculate_rmse(values, fitted)

    def lognormal_fit(climate, environment):
        values = get_environment_radius(environment, climate)
        log_values = np.log(values)
        z = norm.ppf(percentiles)
        A = np.vstack([np.ones_like(z), z]).T
        mu, sigma = np.linalg.lstsq(A, log_values, rcond=None)[0]
        fitted_log = mu + sigma * z
        fitted = np.exp(fitted_log)
        
        return calculate_rmse(values, fitted)

    def gamma_fit(climate, environment):
        values = get_environment_radius(environment, climate)

        def objective(params):
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return np.inf
            est = gamma.ppf(percentiles, a=shape, scale=scale)
            return np.sqrt(np.mean((est - values) ** 2))

        mean = np.mean(values)
        std = np.std(values)
        shape_init = (mean / std) ** 2
        scale_init = (std ** 2) / mean

        res = minimize(objective, [shape_init, scale_init], bounds=[(1e-6, None), (1e-6, None)], method='L-BFGS-B')
        shape_opt, scale_opt = res.x
        fitted = gamma.ppf(percentiles, a=shape_opt, scale=scale_opt)
        return calculate_rmse(values, fitted)

    for env in environments:
        for climate in climates:
            labels.append(f"{env}\n{climate}")
            errors["Normal"].append(normal_fit(climate, env))
            errors["Log-normal"].append(lognormal_fit(climate, env))
            errors["Gamma"].append(gamma_fit(climate, env))

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use softer but distinct colors (colorblind-friendly palette)
    colors = {
        "Normal": "#1f77b4",    # muted blue
        "Log-normal": "#ff7f0e",# orange
        "Gamma": "#2ca02c",     # green
    }

    bars_normal = ax.bar(x - width, errors["Normal"], width, label="Normal", color=colors["Normal"], edgecolor='black', linewidth=0.7)
    bars_lognorm = ax.bar(x, errors["Log-normal"], width, label="Log-normal", color=colors["Log-normal"], edgecolor='black', linewidth=0.7)
    bars_gamma = ax.bar(x + width, errors["Gamma"], width, label="Gamma", color=colors["Gamma"], edgecolor='black', linewidth=0.7)

    # Label formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel("RMSE (km)", fontsize=14)

    # Grid only horizontal lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # Legend
    legend = ax.legend(frameon=True, fontsize=12, loc='upper right', edgecolor='black')

    # Tight layout for clean spacing
    plt.tight_layout()

    # Add RMSE values on top of each bar for clarity
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 6),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

    autolabel(bars_normal)
    autolabel(bars_lognorm)
    autolabel(bars_gamma)
    plt.savefig("fit_errors_pretty.png", dpi=300, bbox_inches='tight')
    plt.show()
    
plot_fit_errors_pretty()