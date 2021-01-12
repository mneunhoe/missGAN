#' @title create_toydata
#'
#' @description Provides an overview table for the time and scope conditions of
#'     a data set
#'
#' @param n The number of observations in the produced toydata
#' @param seed Seed to get reproducible data
#' @return A matrix with two columns and n rows.
#' @examples
#' toydata <- create_toydata(n = 1000)
#' @export

create_toydata <- function(n = 1000, seed = NULL) {

  # If a seed is provided, set it
  if(!is.null(seed)) {
    set.seed(seed)
  }

  # The first variable a is just a draw from a standard normal distribution
  a <- rnorm(n)

  # The second variable is a function of a plus some noise
  b <- rnorm(n, a^2, 0.3)

  # Put a and b together in one matrix
  df <- cbind(a, b)

  # Return the matrix
  return(df)
}
