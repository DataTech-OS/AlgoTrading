install.packages("quantmod")

# get data using Yahoo finance (the ticker for the S&P500 is ^GSCP)
require(quantmod)
getSymbols("^GSPC")

# the return is just the difference in the log of the adjusted closing prices
sp500ret <- diff(log(Ad(GSPC)))

# now we try to determine the best model using the Bayesian information criteria
result.bic <- Inf
result.order <- c(0, 0, 0)
result.arma <- arima(sp500ret, order=result.order)

for (i in 0:4)
{
  for (j in 0:4)
  {
    current.arma = arima(sp500ret, order=c(i, 0, j))
    current.bic = BIC(current.arma)

    if (current.bic < result.bic)
    {
      result.bic <- current.bic
      result.order <- c(i, 0, j)
      result.arma <- current.arma
    }
  }
}

# let's see the final order of the model
result.order

# let's plot the sample autocorrelation of the residuals...
acf(resid(result.arma), na.action=na.omit)

# ...and perform the Ljung-Box test on the residuals
pvalue <- function(model, lags)
{
  tres <- Box.test(resid(model), lag=lags, type="Ljung-Box")
  return(tres$p.value)
}
pval <- pvalue(result.arma, 35)
pval