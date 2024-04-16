
crosscor <- function(x,y, lagmax)
{
  lags <-  seq(-1*lagmax,lagmax)
  corrs <- data.frame()
  for (lag in lags)
  {
    if (lag >= 0)
    {
      cc <- cor.test(x, lag(y,lag))
    }
    else
    {
      cc <- cor.test(lag(x, -1*lag),y)
    }
    corrs <- rbind(corrs, data.frame(lag=lag, corr = cc$estimate, low = cc$conf.int[1], high = cc$conf.int[2]))
  }
  return(corrs)
}

