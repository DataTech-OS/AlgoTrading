install.packages("depmixS4")

require("depmixS4")
require("quantmod")

getSymbols("^GSPC", from="2010-01-01", to="2020-01-01")
sp500ret <- as.numeric(diff(log(Ad(GSPC))))

plot(sp500ret, type="l")

hmm <- depmix(sp500ret~1, data=data.frame(returns=sp500ret), nstates=3)
hmmfit <- fit(hmm, verbose=FALSE)
post_prob <- posterior(hmmfit)

layout(1:2)
plot(post_prob$state, type="s", ylab="State", xlab="Day", ylim=c(1,3.1))
matplot(post_prob[-1], type="l", ylab="Posterior Probability")