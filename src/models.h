#ifndef MODEL_H_
#define MODEL_H_

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// Abstract parent class:
class Model
{
public:
	virtual arma::colvec calculateGradient (arma::colvec& param) const = 0;
	virtual double calculateMSE (arma::colvec& param) const = 0;

  virtual ~Model ()
	{
		std::cout << "Destroy Model" << std::endl;
	}
};

// Linear Model class:
class LinearModel : public Model
{
public:

	LinearModel (arma::mat X, arma::colvec y) : X(X), y(y)
	{
		arma::mat temp = X.t();

		XtX = temp * X;
		Xty = temp * y;
	};

	~LinearModel ()
	{
		std::cout << "Destroy Linear Model" << std::endl;
	}

	arma::colvec calculateGradient (arma::colvec& param) const
	{
		return (2 * (Xty - XtX * param) / X.n_rows);
	};
	double calculateMSE (arma::colvec& param) const
	{
		return (arma::accu(arma::pow(y - X * param, 2)) / X.n_rows);
	};

private:
	arma::mat X;
	arma::colvec y;

	arma::mat XtX;
	arma::mat Xty;
};

#endif // MODEL_H_