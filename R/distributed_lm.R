# ========================================================================================== #
#                                                                                            #
#                      Distributed Learning With Multiple Data Sources                       # 
#                                                                                            #
# ========================================================================================== #

#' Train a linear model using gradient descent
#' 
#' This function is just a wrapper around the \code{lmGradientDescent()} function written
#' in \code{C++}. 
#' 
#' @param formula [\code{formula}]\cr
#'   Formula analog to the formula call in \code{lm}.
#' @param data [\code{data.frame}]\cr
#'   Data frame containing the data used for modeling.
#' @param iters [\code{integer(1)}]\cr
#'   Number of maximal iterations. Could be less if the "epsilon criteria" is hit.
#' @param learning_rate [\code{numeric(1)}]\cr
#'   The step size used for gradient descent. Note: If the mse is not improving the step size
#'   is shrinked by 20 percent.
#' @param beta_init [\code{numeric}]\cr
#'   Initial vector of coefficients used as starting point for the gradient descent.
#' @param mse_eps [\code{numeric(1)}]\cr
#'   Relativ improvement of the MSE. If this boundary is undershot, then the algorithm stops.
#' @param trace [\code{logical(1)}]\cr
#'   Flag if the trace should be printed or not.
#' @returns List of parameter vector, the final mse, and a flag if the algorithm was stopped
#'   by the "epsilon criteria" or after the maximal iterations.
distributedLinearModel = function (formula, param.dir, files, iters, mse_eps)
{
	# Check if files exists:
	check.files = file.exists(files)

	if (any(! check.files)) {
		warning ("Following files does not exist: ", paste(files[check.files], collapse = ", "))
	}



}