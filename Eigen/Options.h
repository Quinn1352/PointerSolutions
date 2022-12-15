#pragma once
#include <Eigen/Dense>

using Eigen::MatrixXd;

struct Options
{
	double rho;
	double epsPrim;
	double epsDual;
	int maxIter;
	MatrixXd xInit;
	MatrixXd zInit;
	MatrixXd yOneInit;
	MatrixXd yTwoInit;
	int method;
	bool useOptions;
};
