#include <RcppCommon.h>

class Test 
{
private:
	double x;
	double y;

public:
	Test (double x, double y) : x ( x ), y ( y ) {};
	double getXtimesY () { return x * y; };

	double getX () const { return x; };
	double getY () const { return y; };
};

  // non-intrusive extension via template specialisation
  template <> Test Rcpp::as(SEXP mytest);
  //
  // non-intrusive extension via template specialisation
  template <> SEXP Rcpp::wrap(const Test& test);

#include <Rcpp.h>

template <> Test Rcpp::as(SEXP testsexp) {
  // Rcpp::List testlist(testsexp);
  // return Test(testlist["x"], testlist["y"]);

  Rcpp::XPtr<Test> ptr(testsexp);
  return *ptr;
}

template <> SEXP Rcpp::wrap(const Test& test) {

  // return Rcpp::wrap(Rcpp::List::create(
  // 	 Rcpp::Named("x") = test.getX(), 
  // 	 Rcpp::Named("y") = test.getY()
  // ));
  Rcpp::XPtr<const Test> ptr(&test);
  return Rcpp::wrap(ptr);
}


// [[Rcpp::export]]
double doSomethingWithTest (SEXP test) {

	Test testobj = Rcpp::as<Test>(test);

	return testobj.getXtimesY();

}

// [[Rcpp::export]]
SEXP getTest (double a, double b)
{
	Test* temp = new Test(a, b);
	Rcpp::XPtr<Test> ptr(temp);
	return Rcpp::wrap(ptr);
}

// // [[Rcpp::export]]
// SEXP foo() {

//   int *a = new int(1);

//   Rcpp::XPtr<int> ptr(a);

//   return ptr;
// }

// // [[Rcpp::export]]
// int bar(SEXP a){

//   Rcpp::XPtr<int> x(a);
//   int b = *x;

//   return b;
// }