# ========================================================================================== #
#                                                                                            #
#                       Gradient Descent Wrapper Around C++ Function                         # 
#                                                                                            #
# ========================================================================================== #

#' Train a linear model using gradient descent
#' 
#' This function is just a wrapper around the \code{lmGradientDescent_internal()} function 
#' written in \code{C++}. 
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
#' @param warnings [\code{logical(1)}]\cr
#'   Flag to specify if warnings should be printed or not.
#' @returns List of parameter vector, the final mse, and a flag if the algorithm was stopped
#'   by the "epsilon criteria" or after the maximal iterations.
lmGradientDescent = function (formula, data, iters = 100L, learning_rate = 0.05, beta_init = NULL, 
	mse_eps = 1e-6, trace = FALSE, warnings = FALSE)
{
	response = all.vars(formula)[attr(terms(formula, data = data), "response")]
	X = model.matrix(formula, data = data)

	if (is.null(beta_init)) {
		beta_init = 0
	}
	out = lmGradientDescent_internal(X, data[[response]], iters = iters, learning_rate = learning_rate,
		beta_init = beta_init, mse_eps = mse_eps, trace = trace, warnings = warnings)

	rownames(out$beta) = colnames(X)
	class(out) = "lmGradDesc"
	return (out)
}

#' Conduct one gradient descent step
#' 
#' This function is just a wrapper around the \code{updateBeta_internal()} function written
#' in \code{C++}. 
#' 
#' @param formula [\code{formula}]\cr
#'   Formula analog to the formula call in \code{lm}.
#' @param data [\code{data.frame}]\cr
#'   Data frame containing the data used for modeling.
#' @param learning_rate [\code{numeric(1)}]\cr
#'   The step size used for gradient descent. Note: If the mse is not improving the step size
#'   is shrinked by 20 percent.
#' @param actual_beta [\code{numeric}]\cr
#'   Actual coefficient vector which should be updated.
#' @param mse_eps [\code{numeric(1)}]\cr
#'   Relativ improvement of the MSE. If this boundary is undershot, then the algorithm stops.
#' @param trace [\code{logical(1)}]\cr
#'   Flag if the trace should be printed or not.
#' @param warnings [\code{logical(1)}]\cr
#'   Flag to specify if warnings should be printed or not.
#' @returns List of parameter vector, the final mse, and a flag if the algorithm was stopped
#'   by the "epsilon criteria" or after the maximal iterations.
updateBeta = function (formula, data, learning_rate = 0.05, actual_beta, mse_eps, 
	trace = FALSE, warnings = FALSE)
{
	response = all.vars(formula)[attr(terms(formula, data = data), "response")]
	X = model.matrix(formula, data = data)

	actual_mse = mean((data[[response]] - X %*% actual_beta)^2)
	out = updateBeta_internal (X, data[[response]], t(X) %*% X, t(X) %*% data[[response]], 
		actual_beta, actual_mse, learning_rate, mse_eps, trace, warnings)

	rownames(out$beta) = colnames(X)
	return (out)
}