#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

namespace model
{
// Abstract parent class:
class Model
{
public:
	virtual arma::colvec calculateGradient (arma::colvec&) const = 0;
	virtual double calculateMSE (arma::colvec&) const = 0;
	virtual arma::colvec predict (arma::mat&) const = 0;
	virtual arma::colvec predict () const = 0;

	arma::colvec getParameter () const;
	arma::colvec setParameter (arma::colvec&);

  virtual ~Model ();

protected:
	arma::colvec parameter;
};


// Linear Model class:
class LinearModel : public Model
{
public:

	LinearModel (arma::mat, arma::colvec);
	~LinearModel ();

	arma::colvec calculateGradient (arma::colvec&) const;
	double calculateMSE (arma::colvec&) const;
	arma::colvec predict (arma::mat&) const;
	arma::colvec predict () const;

private:
	arma::mat X;
	arma::colvec y;

	arma::mat XtX;
	arma::mat Xty;
};

} // namespace model

#endif // MODEL_H_