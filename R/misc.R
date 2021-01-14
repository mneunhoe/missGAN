# Function to generate data according to King et al 2001

tbm_data <-
  function(n = 500,
           missingness = "mcar1",
           complexity = "additive") {
    ## Function to create datasets:


    data.gen  <- function(DGP, n, w.rand.error = TRUE, mean.vec) {
      X1 <- rgamma(n, 8, 2)
      X2 <- rgamma(n, 10, 1)
      X3.5 <-  mvrnorm(n, c(2, 3, 6), diag(c(1.5, 0.5, 3.3)))
      X6.8 <-
        t(rmultinom(n, size = 1, prob = c(1 / 3, 1 / 3, 1 / 3)))
      X9.10 <-
        mvrnorm(n, mu = c(-0.3, 2), Sigma = matrix(c(1.5, 0.685, 0.685, 5.5), 2))
      X11.40 <- mvrnorm(n, mu = mean.vec, Sigma = diag(30))

      ### Create outcomes
      Y  <- switch(
        DGP
        ,
        "additive" = X1  +  X2 + X3.5[, 1] + X9.10[, 2]
        ,
        "multiplicative" =  X1 * X2 * X3.5[, 1] * X9.10[, 2]
        ,
        "add+multip" = 1.5 + (3.2) * X1 + 0.1 * X2 - (2.8) * X9.10[, 2] + 1.2 *
          X1 * X2 -
          1.2 * X1 * X9.10[, 2] - 0.2 * X2 * X9.10[, 2] + 2.4 * X1 *
          X2 * X9.10[, 2]
        ,
        "complicated" =  ifelse(
          X9.10[, 2] < 2.5,
          c(
            X1 - X1 ^ 2 - X2 ^ 2  - 15 * X1 * X2 * X9.10[, 2] + poly(X9.10[, 2], 3, raw =
                                                                       TRUE) %*% c(10,-5, 0.9)
          ),
          1750 + 350 * X9.10[, 2]
        )
      )
      if (w.rand.error) {
        Y  <- Y + sd(Y) * rnorm(n)
      }

      ### Create dataset
      temp.data  <-  data.frame(
        Y = Y
        ,
        X1 = X1
        ,
        X2 = X2
        ,
        X3 = X3.5[, 1]
        ,
        X4 = X3.5[, 2]
        ,
        X5 = X3.5[, 3]
        ,
        X6 = X6.8[, 1]
        ,
        X7 = X6.8[, 2]
        ,
        X8 = X6.8[, 3]
        ,
        X9 = X9.10[, 1]
        ,
        X10 = X9.10[, 2]
      )
      temp.data <- cbind(temp.data, as.data.frame(X11.40))
      return(as.matrix(temp.data))
    }


    #Common mean vector for irrelevant variables
    irr_mean_vec <- sample(2:10, 30, replace = TRUE)
    D <- data.gen(DGP = complexity, n = n, mean.vec = irr_mean_vec)
    ## 100 train datasets

    if (missingness == "complete") {
      return(list(D, D))
    } else if (missingness == "mcar1") {
      U_M <- array(runif(prod(dim(D))), dim(D))
      M <- U_M > 0.19
      M[, 1] <- T
      D_mis <- D
      D_mis[!M] <- NA

      return(list(D, D_mis))

    }

  }





mixed_data <-
  function(n = 1000,
           missingness = "mcar1",
           coefs = c(0, 1, 0,-2)) {
    x <- rnorm(n)
    bin <- rbinom(n, 1, 0.5)

    y <-
      coefs[1] + coefs[2] * x + coefs[3] * bin + coefs[4] * x * bin + rnorm(n, 0, 0.2)

    bimod <- c(rnorm(n / 2, -4), rnorm(n / 2, 4))


    D <- cbind(x, bin, y)



    if (missingness == "complete") {
      return(list(D, D))
    } else if (missingness == "mcar1") {
      U_M <- array(runif(prod(dim(D))), dim(D))
      M <- U_M > 0.19
      M[, 3] <- T
      D_mis <- D
      D_mis[!M] <- NA

      return(list(D, D_mis))

    }


  }


hd_data <- function(n = 500, missingness = "mar1") {
  mus <- rep(0, 5)

  varcov <- matrix(0.8, nrow = 5, ncol = 5)

  diag(varcov) <- 1

  D_latent <- MASS::mvrnorm(n, mus, varcov)

  D <- (D_latent > 0) * 1


  if (missingness == "complete") {
    return(list(D, D))
  } else if (missingness == "mar1") {
    U_M <- array(runif(prod(dim(D))), dim(D))
    U_M[rowSums(D[, 3:5]) > 0,] <- 1
    U_M[, 3:5] <- 1
    M <- U_M > 0.2
    D_mis <- D
    D_mis[!M] <- NA

    return(list(D, D_mis))

  } else if (missingness == "mar2") {
    U_M <- array(runif(prod(dim(D))), dim(D))
    U_M[rowSums(D[, 3:5]) > 0,] <- 1
    U_M[, 3:5] <- 1
    M <- U_M > 0.5
    D_mis <- D
    D_mis[!M] <- NA

    return(list(D, D_mis))

  } else if (missingness == "mar3") {
    U_M <- array(runif(prod(dim(D))), dim(D))
    U_M[rowSums(D[, 3:5]) > 0,] <- 1
    U_M[, 3:5] <- 1
    M <- U_M > 0.8
    D_mis <- D
    D_mis[!M] <- NA

    return(list(D, D_mis))

  }


}



amelia_data <- function(n = 500, missingness = "mcar1") {
  # set.seed(seed)

  mus <- rep(0, 5)
  sds <- rep(1, 5)

  sds_mat <- diag(sds)

  cor_mat <- matrix(
    c(
      1 ,-.12,-.1,
      .5,
      .1,
      -.12,
      1,
      .1,-.6,
      .1,
      -.1,
      .1,
      1,-.5,
      .1,
      .5,-.6,-.5,
      1,
      .1,
      .1,
      .1,
      .1,
      .1,
      1
    ),
    nrow = 5,
    ncol = 5,
    byrow = T
  )

  varcov <- sds_mat %*% cor_mat %*% sds_mat

  D <- MASS::mvrnorm(n, mus, varcov)

  if (missingness == "complete") {
    return(list(D, D))
  } else if (missingness == "mcar1") {
    U_M <- array(runif(prod(dim(D))), dim(D))
    M <- U_M > 0.06
    M[, 4] <- T
    D_mis <- D
    D_mis[!M] <- NA

    return(list(D, D_mis))

  } else if (missingness == "mcar2") {
    U_M <- array(runif(prod(dim(D))), dim(D))
    M <- U_M > 0.19
    M[, 4] <- T

    D_mis <- D

    D_mis[!M] <- NA

    return(list(D, D_mis))
  } else if (missingness == "mar1") {
    U_M <- array(runif(prod(dim(D))), dim(D))

    M <- U_M > 0.06

    M[, 4] <- T

    M[, 2] <- !(D[, 4] < -1 & U_M[, 2] < 0.9)
    M[, 3] <- !(D[, 4] < -1 & U_M[, 3] < 0.9)

    D_mis <- D

    D_mis[!M] <- NA

    return(list(D, D_mis))
  } else if (missingness == "mar2") {
    U_M <- array(runif(prod(dim(D))), dim(D))

    M <- U_M > 0.12

    M[, 4] <- T

    M[, 2] <- !(D[, 4] < -0.4 & U_M[, 2] < 0.9)
    M[, 3] <- !(D[, 4] < -0.4 & U_M[, 3] < 0.9)

    D_mis <- D

    D_mis[!M] <- NA

    return(list(D, D_mis))
  } else if (missingness == "ni") {
    U_M <- array(runif(prod(dim(D))), dim(D))

    M <- U_M > 0.06

    M[, 1] <- D[, 1] > -0.95

    M[, 2] <- !(D[, 4] < -0.52)
    M[, 3] <- !(D[, 3] > 0.48)

    D_mis <- D

    D_mis[!M] <- NA

    return(list(D, D_mis))
  }


}

# Pre process data, i.e. generate mask and swap out NAs for tau
pre_process_missing_data <- function(D, tau = 0) {
  M <- (!is.na(D)) * 1
  D[is.na(D)] <- tau
  return(list(D = D, M = M))
}
