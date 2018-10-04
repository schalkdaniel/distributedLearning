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
	void setParameter (arma::colvec&);

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

// Logistic Regression class:
class LogisticRegression : public Model
{
public:
	LogisticRegression (arma::mat, arma::colvec);
	~LogisticRegression ();

	arma::colvec calculateGradient (arma::colvec&) const;
	double calculateMSE (arma::colvec&) const;
	
	arma::colvec responseFun (arma::colvec&) const;
	
	arma::colvec predict (arma::mat&) const;
	arma::colvec predict () const;


private:
	arma::mat X;
	arma::colvec y;

};

// // Poisson Regression class:
// class PoissonRegression : public Model
// {
// public:
// 	PoissonRegression ();
// 	~PoissonRegression ();

// 	arma::colvec calculateGradient (arma::colvec&) const;
// 	double calculateMSE (arma::colvec&) const;
// 	arma::colvec predict (arma::mat&) const;
// 	arma::colvec predict () const;
// };

} // namespace model

#endif // MODEL_H_
