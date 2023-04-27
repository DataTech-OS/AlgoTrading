install.packages("tseries")

require("tseries")
require("quantmod")

pvalue <- function(residuals, lags)
{
  tres <- Box.test(residuals, lag=lags, type="Ljung-Box")
  return(tres$p.value)
}

getSymbols("^FTSE")

ftseret <- diff(log(Ad(FTSE)))
plot(ftseret)

ftseret <- as.numeric(ftseret)
ftseret <- ftseret[!is.na(ftseret)]

result.order <- c(0,0,0)
result.arima <- arima(ftseret, order=result.order)
result.bic <- Inf

for (p in 1:5) for (d in 0:1) for (q in 1:5)
{
  current.order <- c(p, d, q)
  current.arima <- arima(ftseret, order=current.order)
  current.bic <- BIC(current.arima)
  
  if (current.bic < result.bic)
  {
    result.bic <- current.bic
    result.order <- current.order
    result.arima <- current.arima
  }
}

acf(resid(result.arima))
pval <- pvalue(resid(result.arima), 35)
if (pval <= 0.05)
{
  f <- paste(as.character(result.order), collapse=", ")
  cat("Arima(", f ,") still leaves correlation in the residuals", sep = "")
  
  acf(result.residuals^2)
  
  result.bic <- Inf
  result.order <- c(0,0)

  for (p in 1:5) for (q in 1:5)
  {
    current.order <- c(p,q)
    current.garch <- garch(resid(result.arima), order=current.order, trace=FALSE)
    current.bic <- -2 * logLik(current.garch) + (p + q) * log(length(resid(result.arima)))
    
    if (current.bic < result.bic)
    {
      result.bic <- current.bic
      result.order <- current.order
      result.garch <- current.garch
    }
  }
  
  pval <- pvalue(resid(result.garch), 35)
  if (pval > 0.05)
  {
    g <- paste(as.character(result.order), collapse=", ")
    cat("Arima(", f ,") and Garch(", g, ") seem to be a good fit", sep = "")
  }
}