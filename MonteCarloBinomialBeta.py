import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import scipy.stats as stats

def main():
    alpha = 1
    beta = 1
    n = 50
    z = 10
    alpha_p = alpha + z
    beta_p = beta + n - z
    
    model = pm.Model()
    with model:
        prior = pm.Beta("prior", alpha=alpha, beta=beta)
        likelihood = pm.Binomial("likelihood", n=n, observed=z, p=prior)
        
        start = pm.find_MAP()
        step = pm.Metropolis()
        sample_size = 100000 
        
        blob = pm.sample(sample_size, step, start, random_seed=1)
        sample = blob["prior"]
        
    plt.hist(sample, bins=100, histtype="step", density=True, 
             color="red", label="Posterior MCMC distribution")
    
    x = np.linspace(0, 1, 100)
    plt.plot(x, stats.beta.pdf(x, alpha, beta), 
             label="Prior distribution", color="blue")
    plt.plot(x, stats.beta.pdf(x, alpha_p, beta_p), 
             label="Posterior distribution", color="green")
    
    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    pm.traceplot(blob)
    plt.show()
    
if __name__ == "__main__":
    main()