"""Biased coin from Example 1 in Chapter 2, 'Parameter Estimation I'"""

import numpy as np

from matplotlib import pyplot as plt

# Sample points for numerical integration and plotting
QUAD_PTS = np.linspace(0.0, 1.0, 10000)


def flip_coin(p_heads: float, n_flips: int) -> np.ndarray:
    """Flips the biased coin. True is heads, False is tails.

    args:
        p_heads: Probability of heads
        n_flips: Number of simulated flips

    returns:
        The simulated coin flips
    """ 

    rng = np.random.default_rng()
    samples = rng.uniform(0.0, 1.0, n_flips)

    return np.where(samples < p_heads, True, False)


def uniform() -> np.ndarray:
    """Uniform prior.

    returns:
        Probability of p_heads being true
    """
    return np.ones_like(QUAD_PTS)


def gaussian() -> np.ndarray:
    """Gaussian prior centered around 0.5 with standard deviation of 0.1.

    returns:
        Probability of p_heads being true
    """
    return np.exp(-(QUAD_PTS - 0.5)**2/(2*0.1**2))


def likelihood(fl: bool) -> float:
    """Likelihood of a single flip being heads.

    args:
        fl: Result of a single flip

    returns:
        Likelihood
    """

    if fl:
        return QUAD_PTS
    
    return 1.0 - QUAD_PTS


def normalize(dist: np.ndarray) -> np.ndarray:
    """Normalizes the probability distribution.

    args:
        dist: Non-normalized distribution

    returns:
        Normalized distribution
    """
    return dist/np.trapezoid(dist, QUAD_PTS)


def update_posterior(dist: np.ndarray, fl: bool) -> np.ndarray:
    """Updates and normalizes the posterior distribution.

    args:
        dist: Posterior distribution
        fl: Result of a single flip

    returns:
        The updated posterior distribution
    """
    return normalize(dist*likelihood(fl))


def most_likely(dist: np.ndarray) -> tuple[float, float]:
    """Finds the most likely value and its uncertainty.

    args:
        dist: Normalized distribution

    returns:
        Most likely value, uncertainty
    """

    # Location of maximum probability
    loc = np.argmax(dist)

    # Most likely value
    ml = QUAD_PTS[loc]

    # Uncertainty is related to second derivative of logarithm
    dist[dist == 0] += 1e-16
    log_dist = np.log(dist)

    d2 = np.gradient(np.gradient(log_dist, QUAD_PTS, edge_order=2),
                     QUAD_PTS, edge_order=2)

    std_dev = 1.0/np.sqrt(-d2[loc])

    return ml, std_dev


def simulate(prior: np.ndarray, p_heads: float, n_flips: int,
             figtitle: str = None) -> None:
    """Simulates the coin flips, plots the prior and posterior, prints data.

    args:
        prior: Prior distribution
        p_heads: Probability of heads
        n_flips: Number of flips to simulate
        figtitle: Figure title
    """

    prior = normalize(prior)

    for n, fl in enumerate(flip_coin(p_heads, n_flips)):

        if n == 0:
            posterior = update_posterior(prior, fl)
        else:
            posterior = update_posterior(posterior, fl)

    ml, std_dev = most_likely(posterior)

    print(f"Number of flips: {n_flips}\n"
          f"Exact: {p_heads}\n"
          f"Most likely: {ml:.3f}\n"
          f"Uncertainty: {std_dev:.3f}")

    plt.figure(figtitle)
    plt.plot(QUAD_PTS, prior, label="Prior", color='red')
    plt.plot(QUAD_PTS, posterior, label="Posterior", color='blue')
    plt.axvline(p_heads, color='black', ls='--', label='Exact')
    plt.axvspan(ml - 2*std_dev, ml + 2*std_dev, color='gray', alpha=0.3,
                label="95% C.I.")
    plt.xlabel('h')
    plt.xlim([0.0, 1.0])
    plt.ylabel('p(h)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Number of flips to simulate
    N = 2000
    # Probability of heads on the biased coin
    H = 0.25

    print("Uniform prior:\n"+
          "-"*80)
    simulate(uniform(), H, N, "uniform")

    print("\nGaussian prior:\n"+
          "-"*80)
    simulate(gaussian(), H, N, "gaussian")
