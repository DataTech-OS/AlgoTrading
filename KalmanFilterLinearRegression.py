import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pykalman import KalmanFilter

def main():
    etfs = ["TLT", "IEI"]
    start_date = "2010-01-01"
    end_date = "2020-01-01"
    
    # download data from Yahoo!Finance
    tlt = pdr.get_data_yahoo(etfs[0], start_date, end_date)
    iei = pdr.get_data_yahoo(etfs[1], start_date, end_date)
   
    # create dataframe
    df = pd.DataFrame(index=tlt.index)
    df[etfs[0]] = tlt["Adj Close"]
    df[etfs[1]] = iei["Adj Close"]
    
    # plot the data outlining the different dates
    c = np.linspace(0.1, 1, len(df))
    cmap = plt.cm.get_cmap("YlOrRd")
    sp = plt.scatter(df["TLT"], df["IEI"], edgecolor="k", 
                     c=c ,cmap=cmap, alpha=0.8)
    cb = plt.colorbar(sp)
    cb.ax.set_yticklabels([p.date() for p in df[::len(df)//9].index])
    plt.xlabel(etfs[0])
    plt.ylabel(etfs[1])
    plt.show()
    
    # Linear Regression through Kalman Filter
    obs_mat = np.vstack([df[etfs[0]], np.ones(df[etfs[0]].shape)])
    obs_mat = obs_mat.T[:, np.newaxis]

    kf = KalmanFilter(transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      transition_covariance=np.eye(2),
                      observation_covariance=1.0,
                      initial_state_mean=np.zeros(2),
                      initial_state_covariance=np.ones((2,2)),
                      n_dim_state=2,
                      n_dim_obs=1)

    state_means, state_varcov = kf.filter(df[etfs[1]].values)

    # plot the changing intercept and slope of the regression
    pd.DataFrame(dict(slope=state_means[:,0], intercept=state_means[:,1]),
                 index=df.index).plot(subplots=True)
    plt.show()
    
if __name__ == "__main__":
    main()