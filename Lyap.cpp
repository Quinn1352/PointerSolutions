#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

MatrixXd Lyap(MatrixXd A, MatrixXd Q)
{
	MatrixXd I(A.rows(), A.cols());
	I = MatrixXd::Identity(I.rows(), I.cols());

	VectorXd vecQ(Map<VectorXd>(Q.data(), Q.cols() * Q.rows()));

	MatrixXd kronOne = Kron(I, A);
	MatrixXd kronTwo = Kron(A, I);
	MatrixXd kronSum = kronOne + kronTwo;

	VectorXd vecX = -(kronSum.householderQr().solve(vecQ));

	Map<MatrixXd> X(vecX.data(), vecX.rows(), vecX.cols());

	return X;
}

MatrixXd Kron(MatrixXd A, MatrixXd B)
{
	MatrixXd C(A.rows() * B.rows(), A.cols() * B.cols());

	for (int i = 0; i < A.cols(); i++) {
		for (int j = 0; j < A.rows(); j++) {
			C.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
		}
	}

	return C;
}