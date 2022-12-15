#pragma once
#include <Eigen/Dense>

using Eigen::MatrixXd;

struct Output
{
	MatrixXd X;
	MatrixXd Z;
	MatrixXd YOne;
	MatrixXd YTwo;
	double Jp;
	double Jd;
	double Rp;
	double dg;
	int steps;
	int timer;
	bool flag;
};
