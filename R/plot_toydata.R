#' @title plot_toydata
#'
#' @description Provides an overview table for the time and scope conditions of
#'     a data set
#'
#' @param dat The number of observations in the produced toydata
#' @param col Set your favorite color (if you don't like the default)
#' @param ... Additional arguments passed to plot
#' @return A plot of the toydata
#' @examples
#' toydata <- create_toydata(n = 1000)
#' plot_toydata(toydata)
#' @export

plot_toydata <- function(dat, col = viridis::viridis(2, alpha = 0.7)[1], pch = 19, ...) {
  plot(
    dat,
    col = col,
    pch = pch,
    bty = "n",
    xlab = "Var 1",
    ylab = "Var 2",
    main = paste0("Toydata with ", nrow(toydata), " observations."),
    las = 1,
    ...
  )
}
