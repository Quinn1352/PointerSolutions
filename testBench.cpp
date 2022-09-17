#include <iostream>
#include <iomanip>
#include <armadillo>
#include <math.h>
#include "Lyap.h"
#include "Options.h"
#include "Output.h"
#include "CCAMA.h"

using namespace std;
using namespace arma;

/*
int main() {

	//number of masses
	int N = 2;
    int dispW = 20;

	//identity and zero matrices
	mat I(N, N, fill::eye);
	mat Ibig(N * 2, N * 2, fill::eye);
	mat ZZ(N, N, fill::zeros);

	//state and input matrices
	vec temp(N, fill::zeros);
	temp(0) = 2;
	temp(1) = -1;
	mat T = toeplitz(temp);

	//dynamic matrix
	//A = [ZZ, I; -T, -I]
	mat A(2 * N, 2 * N);
	mat TEMP1 = join_rows(ZZ, I);
	mat TEMP2 = join_rows(-T, -I);
    A = join_cols(TEMP1, TEMP2);


	//input matrix
	//B = [ZZ; I]
	mat B = join_cols(ZZ, I);

	//output matrix
	mat C = Ibig;

	//dynamics of the filter that generates colored noise
	mat Af = -I;
	mat Bf = I;
	mat Cf = I;
	mat Df = ZZ;

	//from cascade connection of plant and filter
	//Ac = [A, B*Cf; zeros(N,2*N), Af];

	mat TEMP = B * Cf;
	TEMP1 = join_rows(A, TEMP);
	mat TEMP3 = zeros(N, 2 * N);
	TEMP2 = join_rows(TEMP3, Af);
	mat Ac = join_cols(TEMP1, TEMP2);


	//Bc = [B*Df; Bf];
	TEMP = B * Df;
	mat Bc = join_cols(TEMP, Bf);
	//Lyapunov equation for covariance matrix of the cascade systems
	TEMP = Bc * Bc.t();
	mat P(3 * N, 3 * N);
	P = lyap(Ac, TEMP);


	//covariance of the state of the plant
	mat Sigma(2 * N, 2 * N);
	for (int i = 0; i < (2 * N); i++) {
		for (int j = 0; j < (2 * N); j++) {
			Sigma(i, j) = P(i, j);
		}
	}
    

	//structural identity matrix for known elements of covariance matrix Sigma
	TEMP1 = join_rows(I, I);
	mat E = join_cols(TEMP1, TEMP1);

	//matrix of known output correlations
	mat G = E % Sigma;
    
    

	//low rank parameter gamma
	int gamma = 10;

	//input options into the optimization procedure
	Options options;

	options.rho = 10;
	options.epsPrim = pow(10, -6);
	options.epsDual = pow(10, -6);
	options.maxIter = pow(10, 5);

	//initial conditions
	mat Xinit = lyap(A, Ibig);
	options.xInit = Xinit;
	options.zInit = Ibig;

	mat Y1init = lyap(A.t(), -Xinit);
	double tempNorm = norm(Y1init, 2);
	options.yOneInit = (gamma * Y1init) / tempNorm;
	int n = C.n_rows;
	int m = C.n_cols;
	mat Y2Init(n, n, fill::eye);
	options.yTwoInit = Y2Init;
	options.method = 1;


	///////////////////////////////////////////START CCAMA////////////////////////////////////////////////////////////////////////



    //declarations needed for output
    Output output;
    mat Y1new;
    mat Y2new;
    mat Xnew;
    mat Znew;

    //define identity matrices of size m and n for use throughout program
    mat I_m(m, m, fill::eye);
    mat I_n(n, n, fill::eye);


    //Data Preprocessing 
    double rho0 = options.rho;
    double eps_prim = options.epsPrim;
    double eps_dual = options.epsDual;
    int MaxIter = options.maxIter;
    mat X = options.xInit;
    mat Z = options.xInit;
    mat Y1 = options.yOneInit;
    mat Y2 = options.yTwoInit;


    //AMA implementation
    if (options.method == 1) {

        vec eigLadY = real(eig_gen(A.t() * Y1 + Y1 * A + C.t() * (E % Y2) * C));   //% is element wise multiplication
        double logdetLadY = sum(log(eigLadY));
        cx_double traceGY2 = trace(G * Y2);


        cx_double dualY = logdetLadY - trace(G * Y2) + m;

        // identity matrix
        mat Ibig = I_m;
        // backtracking parameters
        double beta = 0.5;
        // initial step size
        double rho1 = rho0;
        vector<int> bbfailures;   //empty 0x0 matrix; not sure the use; come back to

        //print function to print all matrix values needs to go here
        cout << left;
        cout << setw(dispW) << "Rho_BB" << setw(dispW) << "rho" << setw(dispW) << "eps_prim" << setw(dispW) << "res_prim" << setw(dispW) << "eps_dual" << setw(dispW) << "abs(eta)" << setw(dispW) << "iter" << setw(dispW) << "testRan" << endl;

        //Timer should go here


        //Use AMA to solve the gamma parametized problem
        int AMAstep = 0;
        for (AMAstep = 1; AMAstep <= MaxIter; AMAstep++) {

            //X minimization step
            Xnew = solve(A.t() * Y1 + Y1 * A + C.t() * (E % Y2) * C, Ibig);
            Xnew = (Xnew + Xnew.t()) / 2;
            

            vec eigX = real(eig_gen(Xnew));
            double logdetX = sum(log(eigX));

            //Gradient of the dual function
            mat gradD1 = A * Xnew + Xnew * A.t();
            mat gradD2 = (E % (C * Xnew * C.t())) - G;

            mat Rnew2 = gradD2;

            double rho = rho1;

            //declarations moved outside for loop so they can be called later

            vec Svecnew;
            cx_double dualYnew;
            mat Rnew1;
            int testRan = 0;


            for (int j = 1; j <= 50; j++) {

                testRan++;

                double a = gamma / rho;

                //Z minimization step
                mat Wnew = -(A * Xnew + Xnew * A.t() + (1 / rho) * Y1);
                mat U; vec s; mat V;    //declare svd variables
                svd(U, s, V, Wnew);     //svd of Wnew stored in U, s, and V
                vec Svec = s;           //SVD is X = U*S*V' where S is diag matrix with s on its diags
                //thus no need for line Svec = diag(S) from MatLab

//singular value thresholding
                Svecnew = ((1 - a / abs(Svec)) % Svec) % (abs(Svec) > a);

                //update Z
                Znew = U * diagmat(Svecnew) * V.t(); 

                
                Znew = (Znew + Znew.t()) / 2;

                //Y Update
                Rnew1 = gradD1 + Znew;
                //Lagrange multiplier update step
                Y1new = Y1 + rho * Rnew1;
                Y2new = Y2 + rho * Rnew2;
                Y1new = (Y1new + Y1new.t()) / 2;
                Y2new = (Y2new + Y2new.t()) / 2;

                //e-values of Xinv
                mat ladYnew = (A.t() * Y1new) + (Y1new * A) + (C.t() * (E % Y2new) * C);
                vec eigladYnew = real(eig_gen((A.t() * Y1new) + (Y1new * A) + (C.t() * (E % Y2new) * C)));

                cx_double logdetLadYnew = log_det(ladYnew);
                double doubleOfM = m;
                dualYnew = logdetLadYnew - trace(G * Y2new) + doubleOfM;


                if (min(eigladYnew) < 0) {
                    rho = rho * beta;
                }
                else if (real(dualYnew) < real(dualY) +
                    trace(gradD1 * (Y1new - Y1)) +
                    trace(gradD2 * (Y2new - Y2)) -
                    (0.5 / rho) * pow(norm(Y1new - Y1, "fro"), 2) -
                    (0.5 / rho) * pow(norm(Y2new - Y2, "fro"), 2)) {
                    rho = rho * beta;
                }
                else
                    break;
            }

            //Quinn's code goes here
            //primal residual


            Rnew1 = (A * Xnew) + (Xnew * A.t()) + Znew; 
            Rnew2 = E % (C * Xnew * C.t()) - G;
            

            double normRnew1 = norm(Rnew1, "fro");
            normRnew1 = pow(normRnew1, 2);
            double normRnew2 = norm(Rnew2, "fro");
            normRnew2 = pow(normRnew2, 2);
            double sumNorms = normRnew1 + normRnew2;
            double res_prim = sqrt(sumNorms);


            //calculating the duality gap
            cx_double eta = -logdetX + (gamma * sum(Svecnew) - dualYnew);

            if (AMAstep % 1 == 0) {
                cout << setw(dispW) << rho1 << setw(dispW) << rho << setw(dispW) << eps_prim << setw(dispW) << res_prim << setw(dispW) <<
                    eps_dual << setw(dispW) << abs(eta) << setw(dispW) << AMAstep << setw(dispW) << testRan << endl;
            }

            //BB stepsize selection starts here
            mat Xnew1 = (A.t() * Y1new) + (Y1new * A) + (C.t() * (E % Y2new) * C);
            Xnew1 = solve(Xnew1, Ibig);
            Xnew1 = (Xnew1 + Xnew1.t()) / 2;
            

            //gradient of dual function
            mat gradD1new = (A * Xnew1) + (Xnew1 * A.t());
            mat gradD2new = (E % (C * Xnew1 * C.t())) - G;
            


            //numerator for rho1
            mat changeY1 = Y1new - Y1;
            cx_double changeY1Norm = norm(changeY1, "fro");
            changeY1Norm = pow(changeY1Norm, 2);
            mat changeY2 = Y2new - Y2;
            cx_double changeY2Norm = norm(changeY2, "fro");
            changeY2Norm = pow(changeY2Norm, 2);
            cx_double rho1num = changeY1Norm + changeY2Norm;

            //denominator for rho1
            mat changeGradD1 = gradD1 - gradD1new;
            mat prodChangY1ChangeGradD1 = changeY1 * changeGradD1;
            cx_double traceY1 = trace(prodChangY1ChangeGradD1);
            mat changeGradD2 = gradD2 - gradD2new;
            mat prodChangY2ChangeGradD2 = changeY2 * changeGradD2;
            cx_double traceY2 = trace(prodChangY2ChangeGradD2);
            cx_double rho1den = traceY1 + traceY2;

            cx_double rho1Complex = rho1num / rho1den;
            rho1 = real(rho1Complex);

            mat RHO1(1, 1);
            RHO1(0, 0) = rho1;
            if (rho1 < 0 || RHO1.has_nan()) {
                rho1 = rho0;
                bbfailures.push_back(AMAstep);
            }
            //end bb stepsize selection 

            //record path of relevant primal and dual quantities 

            double detXnew = det(Xnew);
            double logdetXnew = log(detXnew);
            double neglogdetXnew = -1 * logdetXnew;
            vec svdZnew = svd(Znew);
            double nuc_normZnew = sum(svdZnew);
            nuc_normZnew = nuc_normZnew * gamma;

            output.Jp = neglogdetXnew + nuc_normZnew;
            output.Jd = real(dualYnew);
            output.Rp = res_prim;
            output.dg = abs(eta);
            //NEED AN OUTPUT.TIME 


            //various stopping criteria
            if (abs(eta) < eps_dual && res_prim < eps_prim) {
                cout << "AMA converged to assigned accuracy!!!" << endl;
                cout << "AMA Steps: " << AMAstep << endl;
                cout << rho1 << "\t" << rho << "\t" << eps_prim << "\t" << res_prim << "\t" <<
                    eps_dual << "\t" << abs(eta) << "\t" << AMAstep << endl;
                break;
            }

            Y1 = Y1new;
            Y2 = Y2new;
            dualY = dualYnew;
            // 


        }


        if (AMAstep == MaxIter) {
            output.flag = 0;
        }
        else {
            output.flag = 1;
        }

        output.X = Xnew;
        output.Z = Znew;
        output.YOne = Y1new;
        output.YTwo = Y2new;
        output.steps = AMAstep;

    }


}

// */