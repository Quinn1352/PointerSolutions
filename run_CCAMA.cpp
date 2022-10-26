#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <math.h>
#include "Lyap.h"
#include "Options.h"
#include "Output.h"
#include "CCAMA.h"
#include <chrono>
#include <fstream>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace chrono;

int main()
{
	// Initialize parallelization and set number of threads
	Eigen::setNbThreads(2);

	// Number of masses
	int N = 2;

	MatrixXd I(N, N);
	I = MatrixXd::Identity(I.rows(), I.cols());

	MatrixXd Ibig(N * 2, N * 2);
	Ibig = MatrixXd::Identity(Ibig.rows(), Ibig.cols());

	MatrixXd ZZ(N, N);
	ZZ = MatrixXd::Zero(ZZ.rows(), ZZ.cols());

	VectorXd temp(N);
	temp = VectorXd::Zero(N);

	temp(0) = 2;
	temp(1) = -1;
	MatrixXd T = Toeplitz(temp);

	// dynamic matrix
	MatrixXd TEMP1(ZZ.rows(), ZZ.cols() + I.cols());
	TEMP1 << ZZ, I;
	
	MatrixXd TEMP2(-T.rows(), -T.cols() + -I.cols());
	TEMP2 << -T, -I;

	MatrixXd A(TEMP1.rows() + TEMP2.rows(), TEMP1.cols());
	A << TEMP1, TEMP2;

	// input matrix
	MatrixXd B(ZZ.rows() + I.rows(), ZZ.cols());
	B << ZZ, I;

	// output matrix
	MatrixXd C = Ibig;

	// dynamics of the filter that generates colored noise
	MatrixXd Af = -I;
	MatrixXd Bf = I;
	MatrixXd Cf = I;
	MatrixXd Df = ZZ;

	// from cascade connection of plant and filter
	// Ac = [A, B*Cf, zeros(N, 2*N, Af};

	MatrixXd TEMP = B * Cf;

	MatrixXd TEMP3(A.rows(), A.cols() + TEMP.cols());
	TEMP3 << A, TEMP;

	MatrixXd TEMP4(N, 2 * N);
	TEMP4 = MatrixXd::Zero(TEMP4.rows(), TEMP4.cols());

	MatrixXd TEMP5(TEMP3.rows(), TEMP3.cols() + Af.cols());
	TEMP5 << TEMP3, Af;

	MatrixXd Ac(TEMP1.rows() + TEMP2.rows(), TEMP1.cols());
	Ac << TEMP1, TEMP2;

	// Bc = [B*Df; Bf];

	TEMP = B * Df;
	MatrixXd Bc(TEMP.rows() + Bf.rows(), TEMP.cols());
	Bc << TEMP, Bf;

	// Lyapunov equation for covariance matrix of the cascade systems
	TEMP = Bc * Bc.transpose();
	MatrixXd P(3 * N, 3 * N);
	P = Lyap(Ac, TEMP);


	// covariance of the state of the plant
	MatrixXd Sigma(2 * N, 2 * N);

	for (int i = 0; i < (2 * N); i++)
	{
		for (int j = 0; j < (2 * N); j++)
		{
			Sigma(i, j) = P(i, j);
		}
	}

	// structural identity matrix for known elements of covariance matrix Sigma
	MatrixXd TEMP6(I.rows(), I.cols() + I.cols());
	TEMP6 << I, I;

	MatrixXd E(TEMP6.rows() + TEMP6.rows(), TEMP6.cols());
	E << TEMP6, TEMP6;

	// Matrix of known output correlations
	MatrixXd G = E.cwiseProduct(Sigma);

	// low rank parameter gamma
	int gamma = 10;

	// input options into the optimization procedure
	Options options;

	options.rho = 10;
	options.epsPrim = pow(10, -6);
	options.epsDual = pow(10, -6);
	options.maxIter = pow(10, 5);

	// initial conditions
	MatrixXd Xinit = Lyap(A, Ibig);
	options.xInit = Xinit;
	options.zInit = Ibig;

	MatrixXd Y1init = Lyap(A.transpose(), -Xinit);
	double tempNorm = Y1init.norm();
	options.yOneInit = (gamma * Y1init) / tempNorm;

	int n = C.rows();
	int m = C.cols();

	MatrixXd Y2Init(N, N);
	Y2Init = MatrixXd::Identity(Y2Init.rows(), Y2Init.cols());

	options.yTwoInit = Y2Init;
	options.method = 1;

	options.useOptions = true;

	auto tic = high_resolution_clock::now();
	Output output = ccama(A, C, E, G, gamma, n, m, options);
	auto toc = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(toc - tic);
	double time = duration.count() * pow(10, -6);
	cout << fixed << setprecision(4) << "CCAMA execution time was " << time << "seconds" << endl;

	saveData("Xout.csv", output.X);
	saveData("Zout.csv", output.Z);
}

MatrixXd Toeplitz(VectorXd input)
{
	MatrixXd output(input.size(), input.size());

	for (int i = 0; i < input.size(); i++)
	{
		output(i, 0) = input(i);
	}

	for (int i = 0; i < input.size(); i++)
	{
		for (int j = i; j < input.size(); j++)
		{
			output(j, i) = output(i, j);

			if (i < (input.size() - 1) && j < (input.size() - 1))
			{
				output(i + 1, j + 1) = output(i, j);
			}
		}
	}
}

void saveData(string fileName, MatrixXd matrix)
{
	const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");

	ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix.format(CSVFormat);
		file.close();
	}
}