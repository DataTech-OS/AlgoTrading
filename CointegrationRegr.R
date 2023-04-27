require("quantmod")
require("tseries")

pvalue <- function(residuals, lags)
{
  tres <- Box.test(residuals, lag=lags, type="Ljung-Box")
  return(tres$p.value)
}

getSymbols("EWC", from="2010-01-01", to="2020-01-01")
getSymbols("EWA", from="2010-01-01", to="2020-01-01")

ewcadj <- unclass(EWC$EWC.Adjusted)
ewaadj <- unclass(EWA$EWA.Adjusted)

x <- "January 1st 2010 to January 1st 2020"
y <- "Adjusted closing price in USD"

plot(ewcadj, type="l", col="red", xlab=x, ylab=y, main="EWC-EWA", ylim=c(10,30))
par(new=TRUE)
plot(ewaadj, type="l", col="blue", xlab=x, ylab=y, ylim=c(10,30))
par(new=FALSE)

models.first <- lm(ewcadj ~ ewaadj)
models.second <- lm(ewaadj ~ ewcadj)

models.first.test <- adf.test(models.first$residuals, k=1)
models.second.test <- adf.test(models.second$residuals, k=1)

if (models.first.test$p.value < models.second.test$p.value)
{
  beta <- as.numeric(models.first$coefficients[2])
  models.final <- models.first
}
else
{
  beta <- as.numeric(models.second$coefficients[2])
  models.final <- models.second
}

plot(models.first$residuals, type="l")

adf_res <- adf.test(models.first$residuals)
kpss_res <- kpss.test(models.first$residuals)
pp_res <- pp.test(models.first$residuals)
lb_res <- pvalue(models.first$residuals, 35)

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
cat("On the other hand these tests give the opposite result:", ref)