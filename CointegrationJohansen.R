install.packages("urca")

extractCointRel <- function(jotest, maxr)
{
  r <- rep(0, maxr + 1)
  for (i in maxr:1) 
  {
    if (jotest@cval[maxr + i] > jotest@teststat[i])
    {
      r[1] <- maxr - i
      return(r)
    }
  }
  r[1] <- maxr
  for (i in 1:maxr) 
  {
    r[i + 1] <- jotest@V[i]
  }
  return(r)
}

pvalue <- function(residuals, lags)
{
  tres <- Box.test(residuals, lag=lags, type="Ljung-Box")
  return(tres$p.value)
}

test_stationarity <- function(data)
{
  adf_res <- adf.test(data)
  kpss_res <- kpss.test(data)
  pp_res <- pp.test(data)
  lb_res <- pvalue(data, 35)
  
  tests <- c("Augmented Dickey-Fuller test", "KPSS test", "Philipps-Perron test")
  acc <- c()
  ref <- c()
  
  if (adf_res$p.value < 0.05) 
  {
    acc <- c(acc, tests[1])
  } else
  {
    ref <- c(ref, tests[1])
  }
  
  if (kpss_res$p.value > 0.05)
  {
    acc <- c(acc, tests[2])
  } else
  {
    ref <- c(ref, tests[2])
  }
  
  if (pp_res$p.value < 0.05) 
  {
    acc <- c(acc, tests[3])
  } else
  {
    ref <- c(ref, tests[3])
  }
  
  acc <- paste(as.character(acc), collapse=", ")
  ref <- paste(as.character(ref), collapse=", ")
  cat("The final series is stationary according to the following tests:", acc)
  cat("\nOn the other hand these tests give the opposite result:", ref)
  cat("\nThe Ljung-Box test has a p-value of", lb_res)
}

require("quantmod")
require("tseries")
require("urca")

getSymbols("SPY", from="2015-01-01", to="2015-12-31")
getSymbols("IVV", from="2015-01-01", to="2015-12-31")
getSymbols("VOO", from="2015-01-01", to="2015-12-31")

spy <- unclass(SPY$SPY.Adjusted)
ivv <- unclass(IVV$IVV.Adjusted)
voo <- unclass(VOO$VOO.Adjusted)

plot(spy, type="l", col="red", ylim=c(150, 320), ylab="P")
par(new=T)
plot(ivv, type="l", col="blue", ylim=c(150, 320), ylab="P")
par(new=T)
plot(voo, type="l", col="yellow", ylim=c(150, 320), ylab="P")
par(new=F)

jotest <- ca.jo(data.frame(spy, ivv, voo), type="trace", K=2, ecdet="none", spec="longrun")
p = extractCointRel(jotest, 3)
if (p[1] != 0)
{
  final <- spy * p[2] + ivv * p[3] + voo * p[4]
  plot(final, type="l")
  acf(final)
  test_stationarity(final)
}