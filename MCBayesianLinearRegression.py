import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import arviz as az
from scipy.stats import t

def simulate_linear_data(n, beta_0, beta_1, mu, sigma):
    df = pd.DataFrame({"x" : [np.random.rand() for i in range(n)]})
    df["y"] = beta_0 + beta_1 * df["x"] + np.random.normal(mu, sigma, n)
    return df

def bayesian_linear_regression_mcmc(df, n=1000):
    model = pm.Model()
    with model:
        sigma = pm.InverseGamma("sigma", alpha=1, beta=1)
        beta_0 = pm.Normal("alpha", mu=0, sigma=20)
        beta_1 = pm.Normal("beta", mu=0, sigma=20)
        likelihood = pm.Normal("likelihood", mu=beta_0 + beta_1 * df["x"],
                               sigma=sigma, observed=df["y"])
        
        step = pm.NUTS()
        blob = pm.sample(n, step)
    return blob, model

def frequentist_linear_regression(df, n):
    x = np.array(df["x"]).reshape(n, 1)
    y = np.array(df["y"]).reshape(n, 1)
    
    model = lr()
    result = model.fit(x,y)

    return result.intercept_[0], result.coef_[0,0]

def aci(sigma_hat, alpha_f, siglev, degfr):
    ci = []
    m = sigma_hat / np.sqrt(degfr)
    ts = t.ppf(siglev, degfr)
    
    ci.append(alpha_f - (ts * m))
    ci.append(alpha_f + (ts * m))
    
    return ci
              
def bci(sigma_hat, beta_f, sl, degfr, x, n):
    ci = []
    xm = x.mean()
    x_s = sum((x - xm)**2)
    m = np.sqrt(n) * sigma_hat / (np.sqrt(degfr) * x_s)
    ts = t.ppf(sl, degfr)
    
    ci.append(beta_f - (ts * m))
    ci.append(beta_f + (ts * m))
    
    return ci

def main():
    n = 100
    beta_0 = 1
    beta_1 = 2
    mu = 0
    sigma = 0.5
    
    # simulate the data used in the regression
    df = simulate_linear_data(n, beta_0, beta_1, mu, sigma)
    
    # plot the data (do not show yet)
    plt.plot(df["x"], df["y"], "o")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # estimate a frequentist and bayesian linear regression models
    alpha_f, beta_f = frequentist_linear_regression(df, n)
    blob, model = bayesian_linear_regression_mcmc(df)
    
    # let's sample 3 coefficients from the posterior distribution
    ab = pm.sampling.sample_posterior_predictive(blob, samples=3,
                    model=model, var_names=["alpha", "beta"]) 
    # calculate a bayesian estimate for the two coefficients
    alpha_b = blob["alpha"].mean() 
    beta_b = blob["beta"].mean()
    
    # plot the 5 regression lines just obtained
    x = np.linspace(0, 1, 100)
    for alpha, beta in zip(ab["alpha"], ab["beta"]):
        plt.plot(x, alpha + beta * x, color="blue")
    plt.plot(x, alpha_b + x * beta_b, color="red", 
             label="Bayesian", linewidth=3)
    plt.plot(x, alpha_f + x * beta_f, color="green", 
             label="Frequentist")
    plt.legend()
    plt.show()
    
    # plot a few useful stuff
    print(pm.summary(blob).round(2))
    pm.plots.plot_posterior(blob)
    pm.plots.forestplot(blob)
    pm.plots.densityplot(blob) #KDE estimate
    plt.show()
    
    # calculate an HPD credible interval for the coefficients
    alpha_ci = az.hpd(blob["alpha"], credible_interval=0.95)
    beta_ci = az.hpd(blob["beta"], credible_interval=0.95)
    
    # plot the 3 regression lines just obtained
    plt.plot(x, alpha_ci[0] + x * beta_ci[0], color="blue")
    plt.plot(x, alpha_ci[1] + x * beta_ci[1], color="blue")
    plt.plot(x, alpha_b + x * beta_b, color="red", linewidth=3)
    
    # calculate a 95% confidence interval for the 
    # frequentist regression coefficients estimates
    x_i = np.sort(df["x"])
    y_hat = alpha_f + beta_f * x_i
    sigma_hat = y_hat.std()
    alpha_ci_f = aci(sigma_hat, alpha_f, 0.95, n - 2)
    beta_ci_f = bci(sigma_hat, beta_f, 0.95, n - 2, x_i, n)
    
    # plot the 3 regression lines just obtained
    plt.plot(x, alpha_ci_f[0] + x * beta_ci_f[0], color="green")
    plt.plot(x, alpha_ci_f[1] + x * beta_ci_f[1], color="green")
    plt.plot(x, alpha_f + x * beta_f, color="black")
    plt.show()
    
    # predictions and confidence intervals: frequentist case
    x_star = (x_i - x_i.mean())**2
    sxx = sum(x_star)
    se_y = sigma_hat * np.sqrt(1 + (1 / n) + (x_star / sxx))
    y_hat_upper = y_hat + t.ppf(0.95, n - 2) * se_y
    y_hat_lower = y_hat - t.ppf(0.95, n - 2) * se_y

    plt.plot(df["x"], df["y"], "o")
    plt.plot(x, alpha_f + beta_f * x, label="Estimated regression line")
    plt.fill_between(x_i, y_hat_upper, y_hat_lower, alpha=0.5, 
                     color='gray', label="95% interval")
    plt.legend()
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16, rotation=0)
    plt.title("CI plot")
    plt.show()
    
    # predictions and confidence intervals: bayesian case
    ypred = pm.sampling.sample_posterior_predictive(blob, samples=500,
                    model=model)
    y_hat_ci = az.hpd(ypred["likelihood"], credible_interval=0.95)
    y_hat_ci = np.squeeze(y_hat_ci)

    # reshaping and sorting
    idxs = np.array(df["x"]).ravel().argsort()    
    x_ord = np.array(df["x"]).ravel()[idxs]
    
    y = np.array(alpha_b + beta_b * df["x"])
    y_ord = y[idxs]
    y_hat_ci_ord = y_hat_ci[idxs]
    
    # plotting
    plt.plot(df["x"], df["y"], "o")
    plt.plot(x_ord, y_ord, label="Estimated regression line")
    plt.fill_between(x_ord, y_hat_ci_ord[:,0], y_hat_ci_ord[:,1], alpha=0.5, 
                     color='gray', label="95% interval")
    plt.legend()
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16, rotation=0)
    plt.title("CI plot")
    plt.show()
    
if __name__ == "__main__":
    main()