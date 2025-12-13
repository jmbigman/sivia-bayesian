"""Lighthouse problem from Example 3 in Chapter 2, 'Parameter Estimation I'.
Modified to predict both alpha and beta with techniques from Chapter 3."""

import numpy as np

from matplotlib import pyplot as plt

# Assume alpha is known to be between -5 and 5 km, and beta is known to be at
# most 5 km
ALPHA_PTS = np.linspace(-5.0, 5.0, 1000)
BETA_PTS = np.linspace(0.0, 5.0, 500)

ALPHA_GRID, BETA_GRID = np.meshgrid(ALPHA_PTS, BETA_PTS)


def detect_lighthouse(alpha: float, beta: float, n_samples: int) -> np.ndarray:
    """Generate samples of lighthouse detections.

    args:
        alpha: Position on shoreline
        beta: Distance into sea
        n_samples: Number of samples

    returns:
        The positions detected
    """

    # Randomly sample azimuthal angle
    rng = np.random.default_rng()
    theta = rng.uniform(-np.pi/2, np.pi/2, n_samples)

    # Convert to position
    pos = alpha + beta*np.tan(theta)

    return pos


def uniform() -> np.ndarray:
    """Uniform prior.

    returns:
        Probability of points in the interval
    """
    return np.ones_like(ALPHA_GRID)


def marginalize_alpha(dist: np.ndarray) -> np.ndarray:
    """Marginalizes the joint distribution over beta.

    args:
        dist: Distribution

    returns:
        The normalized marginal distribution of alpha
    """
    marg = np.trapezoid(dist, BETA_PTS, axis=0)
    return marg/np.trapezoid(marg, ALPHA_PTS)


def marginalize_beta(dist: np.ndarray) -> np.ndarray:
    """Marginalizes the joint distribution over alpha.

    args:
        dist: Distribution

    returns:
        The normalized marginal distribution of beta
    """
    marg = np.trapezoid(dist, ALPHA_PTS, axis=1)
    return marg/np.trapezoid(marg, BETA_PTS)


def normalize(dist: np.ndarray) -> np.ndarray:
    """Normalizes the probability distribution over the interval.

    args:
        dist: Non-normalized distribution evaluated at the mesh grid

    returns:
        The normalized distribution
    """
    return dist/np.trapezoid(np.trapezoid(dist, ALPHA_PTS, axis=1), BETA_PTS)


def lorentzian(x: float) -> np.ndarray:
    """Lorentzian likelihood for detection of the lighthouse.

    args:
        x: Position detected

    returns:
        Likelihood of the lighthouse position
    """
    return BETA_GRID/(np.pi*(BETA_GRID**2 + (x - ALPHA_GRID)**2))


def update_posterior(dist: np.ndarray, x: float) -> np.ndarray:
    """Updates and normalizes the posterior distribution.

    args:
        dist: Posterior distribution
        x: Position detected

    returns:
        The updated posterior distribution
    """
    return normalize(dist*lorentzian(x))


def most_likely(dist: np.ndarray) -> tuple[float, float, float, float, float]:
    """Finds the most likely values and their uncertainty.

    args:
        dist: Normalized distribution

    returns:
        Most likely alpha, most likely beta, uncertainty in alpha,
        uncertainty in beta, covariance
    """

    # Location of maximum probability
    loc = np.unravel_index(np.argmax(dist), dist.shape)

    # Most likely values
    ml_alpha = ALPHA_GRID[loc]
    ml_beta = BETA_GRID[loc]

    # Uncertainties are related to second derivatives of logarithm
    dist[dist == 0] += 1e-16
    log_dist = np.log(dist)

    d2_aa = np.gradient(np.gradient(log_dist, ALPHA_PTS, axis=1),
                        ALPHA_PTS, axis=1)[loc]

    d2_bb = np.gradient(np.gradient(log_dist, BETA_PTS, axis=0),
                        BETA_PTS, axis=0)[loc]

    d2_ab = np.gradient(np.gradient(log_dist, ALPHA_PTS, axis=1),
                        BETA_PTS, axis=0)[loc]

    common = d2_aa*d2_bb - d2_ab**2

    sigma_aa = np.sqrt(-d2_bb/common)
    sigma_bb = np.sqrt(-d2_aa/common)

    sigma_sq_ab = d2_ab/common

    return ml_alpha, ml_beta, sigma_aa, sigma_bb, sigma_sq_ab


def simulate(alpha: float, beta: float, n_samples: int) -> None:
    """Simulates the lighthouse detections, plots the posterior,
    prints data.

    args:
        alpha: Position on shoreline
        beta: Distance into sea
        n_samples: Number of samples
    """

    prior = normalize(uniform())

    posterior = prior

    for x in detect_lighthouse(alpha, beta, n_samples):

        posterior = update_posterior(posterior, x)

    ml_alpha, ml_beta, std_dev_a, std_dev_b, cov = most_likely(posterior)

    print(f"Number of detections: {n_samples}\n"
          f"Exact alpha: {alpha}\n"
          f"Exact beta: {beta}\n"
          f"Most likely alpha: {ml_alpha:.3f}\n"
          f"Most likely beta: {ml_beta:.3f}\n"
          f"Uncertainty in alpha: {std_dev_a:.3f}\n"
          f"Uncertainty in beta: {std_dev_b:.3f}\n"
          f"Covariance: {cov:.3e}")

    # 95% confidence interval
    alpha_l = ml_alpha - 2*std_dev_a
    alpha_r = ml_alpha + 2*std_dev_a
    beta_l = ml_beta - 2*std_dev_b
    beta_r = ml_beta + 2*std_dev_b

    plt.figure("marginal_alpha")
    plt.plot(ALPHA_PTS, marginalize_alpha(posterior), label="Posterior")
    plt.axvline(alpha, color='black', ls='--', label="Exact")
    plt.axvspan(alpha_l, alpha_r, color='gray', alpha=0.3, label="95% C.I.")
    plt.legend()
    plt.grid()
    plt.xlabel("alpha")
    plt.ylabel("p(alpha)")
    plt.tight_layout()
    plt.show(block=False)

    plt.figure("marginal_beta")
    plt.plot(BETA_PTS, marginalize_beta(posterior), label="Posterior")
    plt.axvline(beta, color='black', ls='--', label="Exact")
    plt.axvspan(beta_l, beta_r, color='gray', alpha=0.3, label="95% C.I.")
    plt.legend()
    plt.grid()
    plt.xlabel("beta")
    plt.ylabel("p(beta)")
    plt.tight_layout()
    plt.show(block=False)

    plt.figure("posterior")
    plt.contourf(ALPHA_GRID, BETA_GRID, posterior, cmap='Reds')
    plt.plot([alpha], [beta], marker='x', color='black', markersize=6,
             linestyle='', label='Exact')
    plt.plot([alpha_l, alpha_r, alpha_r, alpha_l, alpha_l],
             [beta_l, beta_l, beta_r, beta_r, beta_l], color='black',
             label="95% C.I.")
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # True values
    ALPHA = 0.0
    BETA = 2.5
    # Number of detections to simulate
    N = 1000

    simulate(ALPHA, BETA, N)
