#pragma once
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

MatrixXd Lyap(MatrixXd, MatrixXd);
MatrixXd Kron(MatrixXd, MatrixXd);
