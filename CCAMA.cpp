#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <math.h>
#include "Lyap.h"
#include "Options.h"
#include "Output.h"

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;

Output ccama(MatrixXd A, MatrixXd C, MatrixXd E, MatrixXd G,
	int gamma, int n, int m, Options options)
{
	int dispW = 20;

	// declarations needed for output
	Output output;
	MatrixXd Y1new;
	MatrixXd Y2new;
	MatrixXd Xnew;
	MatrixXd Znew;

	// define identity matricies of size m and n for use throughout program
	MatrixXd I_m(m, m);
	I_m = MatrixXd::Identity(I_m.rows(), I_m.cols());

	MatrixXd I_n(n, n);
	I_n = MatrixXd::Identity(I_n.rows(), I_n.cols());

	if (!options.useOptions)
	{
		options.rho = 10;
		options.epsPrim = 1.e-5;
		options.epsDual = 1.e-4;
		options.maxIter = 1.e5;
		MatrixXd Xinit = Lyap(A, I_m);
		options.xInit = Xinit;
		options.zInit = I_m;
		MatrixXd Y1init = Lyap(A.transpose(), -Xinit);
		double tempNorm = Y1init.norm();
		options.yOneInit = (gamma * Y1init) / tempNorm;
		options.yTwoInit = I_n;
		options.method = 1;
	}

	// default options parameters if no options are given


	// Data processing
	double rho0 = options.rho;
	double eps_prim = options.epsPrim;
	double eps_dual = options.epsDual;
	int MaxIter = options.maxIter;
	MatrixXd X = options.xInit;
	MatrixXd Z = options.zInit;
	MatrixXd Y1 = options.yOneInit;
	MatrixXd Y2 = options.yTwoInit;


	// AMA implementation
	if (options.method == 1)
	{
		MatrixXd tempEigenSolver = A.transpose() * Y1 + Y1 * A + C.transpose() * E.cwiseProduct(Y2) * C;
		VectorXd compEigLadY = tempEigenSolver.eigenvalues();
		VectorXd eigLadY = compEigLadY.real();
		ArrayXXd arrayTemp = eigLadY.array().log();
		double logdetLadY = arrayTemp.matrix().sum();
		MatrixXd traceMatrix = G * Y2;
		complex<double> dualY = logdetLadY - traceMatrix.trace() + m;

		// identity matrix
		MatrixXd Ibig = I_m;

		// backtracing parameters
		double beta = 0.5;

		// initial step size
		double rho1 = rho0;
		vector<int> bbfailures;

		cout << left;
		cout << setw(dispW) << "Rho_BB" << setw(dispW) << "rho" << setw(dispW) << "eps_prim"
			<< setw(dispW) << "res_prim" << setw(dispW) << "eps_dual" << setw(dispW) << "abs(eta)"
			<< setw(dispW) << "iter" << endl;

		// Timer should go here


		// Use AMA to solve the gamma parametized problem
		int AMAstep = 0;
		for (AMAstep = 1; AMAstep <= MaxIter; AMAstep++)
		{
			MatrixXd solveTemp = A.transpose() * Y1 + Y1 * A + C.transpose() * E.cwiseProduct(Y2) * C;
			Xnew = solveTemp.householderQr().solve(Ibig);
			Xnew = (Xnew + Xnew.transpose()) / 2;

			VectorXd eigX = Xnew.eigenvalues().real();
			double logdetX = eigX.log().sum();

			// Gradient of the dual function
			MatrixXd gradD1 = A * Xnew + Xnew * A.transpose();
			MatrixXd gradD2 = E.cwiseProduct(C * Xnew * C.transpose()) - G;

			MatrixXd Rnew2 = gradD2;

			double rho = rho1;

			// declaraitions moved outside for loop so they can be called later

			VectorXd Svecnew;
			complex<double> dualYnew;
			MatrixXd Rnew1;

			for (int j = 1; j <= 50; j++)
			{
				double a = gamma / rho;

				// Z minimization step
				MatrixXd Wnew = -(A * Xnew + Xnew * A.transpose() + (1 / rho) * Y1);
				MatrixXd U; VectorXd s; MatrixXd V;		// declare svd variables
				BDCSVD<MatrixXd> svd(Wnew, ComputeFullV | ComputeFullU);
				U = svd.matrixU();
				V = svd.matrixV();
				s = svd.singularValues();
				VectorXd Svec = s;


				// singular value thresholding
				VectorXd absSvec = Svec.cwiseAbs();
				VectorXd invSvec = absSvec.cwiseInverse();
				VectorXd multSvec = -(a * invSvec);
				multSvec += VectorXd::Ones(multSvec.rows());

				ArrayXd comparison = (Svec.cwiseAbs().array() > a).cast<double>();
				Svecnew = (multSvec.cwiseProduct(Svec)).cwiseProduct(comparison.matrix());

				// update Z
				Znew = U * Svecnew.diagonal() * V.transpose();
				Znew = (Znew + Znew.transpose()) / 2;

				// update Y
				Rnew1 = gradD1 + Znew;

				// Lagrange mulitplier update step
				Y1new = Y1 + rho * Rnew1;
				Y2new = Y2 + rho * Rnew2;
				Y1new = (Y1new + Y1new.transpose()) / 2;
				Y2new = (Y2new + Y2new.transpose()) / 2;

				// e-values of Xinv
				MatrixXd ladYnew = (A.transpose() * Y1new) + (Y1new * A) + (C.transpose() * E.cwiseProduct(Y2new) * C);
				VectorXd eigladYnew = ladYnew.eigenvalues().real();

				complex<double> logdetLadYnew = log(ladYnew.determinant());
				double doubleOfM = m;
				dualYnew = logdetLadYnew - (G * Y2new).trace() + doubleOfM;


				if (eigladYnew.minCoeff() < 0)
				{
					rho = rho * beta;
				}

				else if (dualYnew.real() < dualY.real() +
					(gradD1 * (Y1new - Y1)).trace() +
					(gradD2 * (Y2new - Y2)).trace() -
					(0.5 / rho) * pow((Y1new - Y1).norm(), 2) -
					(0.5 / rho) * pow((Y2new - Y2).norm(), 2))
				{
					rho = rho * beta;
				}

				else
					break;
			}

			// Quinn's code goes here
			// primal residual

			Rnew1 = (A * Xnew) + (Xnew * A.transpose()) + Znew;
			Rnew2 = E.cwiseProduct(C * Xnew * C.transpose()) - G;
			double normRnew1 = Rnew1.norm();
			normRnew1 = pow(normRnew1, 2);
			double normRnew2 = Rnew2.norm();
			normRnew2 = pow(normRnew2, 2);
			double sumNorms = normRnew1 + normRnew2;
			double res_prim = sqrt(sumNorms);


			// calculating the duality gap
			complex<double> eta = -logdetX + (gamma * Svecnew.sum() - dualYnew);

			if (AMAstep % 100 == 0)
			{
				cout << setw(dispW) << rho1 << setw(dispW) << rho << setw(dispW) << eps_prim
					<< setw(dispW) << res_prim << setw(dispW) << eps_dual << setw(dispW) << abs(eta)
					<< setw(dispW) << AMAstep << endl;
			}

			// BB stepsize selection starts here
			MatrixXd Xnew1 = (A.transpose() * Y1new) + (Y1new * A) + (C.transpose() * E.cwiseProduct(Y2new) * C);
			Xnew1 = Xnew1.householderQr().solve(Ibig);
			Xnew1 = (Xnew1 + Xnew1.transpose()) / 2;

			// gradient of dual function
			MatrixXd gradD1new = (A * Xnew1) + (Xnew1 * A.transpose());
			MatrixXd gradD2new = E.cwiseProduct(C * Xnew1 * C.transpose()) - G;

			// numerator for rho1
			MatrixXd changeY1 = Y1new - Y1;
			complex<double> changeY1Norm = changeY1.norm();
			changeY1Norm = pow(changeY1Norm, 2);
			MatrixXd changeY2 = Y2new - Y2;
			complex<double> changeY2Norm = changeY2.norm();
			changeY2Norm = pow(changeY2Norm, 2);
			complex<double> rho1num = changeY1Norm + changeY2Norm;

			// denominator for rho1
			MatrixXd changeGradD1 = gradD1 - gradD1new;
			MatrixXd prodChangY1ChangeGradD1 = changeY1 * changeGradD1;
			complex<double> traceY1 = prodChangY1ChangeGradD1.trace();
			MatrixXd changeGradD2 = gradD2 - gradD2new;
			MatrixXd prodChangY2ChangeGradD2 = changeY2 * changeGradD2;
			complex<double> traceY2 = prodChangY2ChangeGradD2.trace();
			complex<double> rho1den = traceY1 + traceY2;

			complex<double> rho1Complex = rho1num / rho1den;
			rho1 = rho1Complex.real();

			if (rho1 < 0 || isnan(rho1))
			{
				rho1 = rho0;
				bbfailures.push_back(AMAstep);
			}
			// end bb stepsize selection

			// record path of relevant primal and dual quantities

			double detXnew = Xnew.determinant();
			double logdetXnew = log(detXnew);
			double neglogdetXnew = -1 * logdetXnew;
			BDCSVD<MatrixXd> svdZ(Znew, ComputeFullV | ComputeFullU);
			VectorXd svdZnew = svdZ.singularValues();
			double nuc_normZnew = svdZnew.sum();
			nuc_normZnew = nuc_normZnew * gamma;

			output.Jp = neglogdetXnew + nuc_normZnew;
			output.Jd = dualYnew.real();
			output.Rp = res_prim;
			output.dg = abs(eta);
			// NEED AN OUTPUT.TIME


			// various stopping criteria
			if (abs(eta) < eps_dual && res_prim < eps_prim)
			{
				cout << "AMA converged to assinged accuracy!!!" << endl;
				cout << "AMA Steps: " << AMAstep << endl;
				cout << setw(dispW) << rho1 << setw(dispW) << rho << setw(dispW) << eps_prim
					 << setw(dispW) << res_prim << setw(dispW) << eps_dual << setw(dispW) << abs(eta)
					 << setw(dispW) << AMAstep << endl;

				break;
			}

			Y1 = Y1new;
			Y2 = Y2new;
			dualY = dualYnew;


		}


		if (AMAstep == MaxIter)
		{
			output.flag = 0;
		}

		else
		{
			output.flag = 1;
		}

		output.X = Xnew;
		output.Z = Znew;
		output.YOne = Y1new;
		output.YTwo = Y2new;
		output.steps = AMAstep;

	}

	return output;
}