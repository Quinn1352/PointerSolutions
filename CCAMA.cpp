#include <iostream>
#include <armadillo>
#include <math.h>
#include "Lyap.h"
#include "Options.h"
#include "Output.h"

using namespace std;
using namespace arma;


int ccama (mat A, mat C, mat E, mat G, int gamma, int n, int m, Options options) 
{       
    //define identity matrices of size m and n for use throughout program
    mat I_m(m, m, fill::eye);
    mat I_n(n, n, fill::eye);

    /*This version of the code only allows for a 7 parameter ccama call meaning we use
      default options. code to determine the number of arguments will need to go hear to determine
      if there are 8 arguments (customs options) or 7 arguments (default options)*/

    //default options parameters if no options are given
    options.rho = 10;
    options.epsPrim = 1.e-5;
    options.epsDual = 1.e-4;
    options.maxIter = 1.e5;
    mat Xinit = lyap (A, I_m);
    options.xInit = Xinit;
    options.zInit = I_m;
    mat Y1init = lyap(A.t(), -Xinit);       //.t() is transponse of that matrix
    options.yOneInit = (gamma * Y1init) / (norm (Y1init, 2));
    options.yTwoInit = I_n;
    options.method = 1;

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
    if(options.method == 1) {

           vec eigLadY = real(eig_gen(A.t() * Y1 + Y1 * A + C.t() * (E%Y2) * C));   //% is element wise multiplication
           double logdetLadY = sum(log(eigLadY));
           double dualY = logdetLadY - trace(G * Y2) + m;

            // identity matrix
            mat Ibig = I_m;
            // backtracking parameters
            double beta = 0.5;
            // initial step size
            double rho1 = rho0;
            mat bbfailures();   //empty 0x0 matrix; not sure the use; come back to

            /*print function to print all 
              matrix values needs to go hear*/

            /*Timer should go here*/


            //Use AMA to solve the gamma parametized problem
            for (int AMAstep = 1; AMAstep <= MaxIter; AMAstep++) {

                //X minimization step
                mat Xnew = solve(A.t() * Y1 + Y1 * A + C.t() * (E % Y2) * C, Ibig);
                Xnew = (Xnew + Xnew.t()) / 2;

                vec eigX = real(eig_gen(Xnew));
                double logdetX = sum(log(eigX));
                
                //Gradient of the dual function
                mat gradD1 = A * Xnew + Xnew * A.t();
                mat gradD2 = (E % (C * Xnew * C.t())) - G;

                mat Rnew2 = gradD2;

                double rho = rho1;

                for (int j = 1; j <= 50; j++) {

                    int a = gamma / rho;

                    //Z minimization step
                    mat Wnew = -(A * Xnew + Xnew * A.t() + (1 / rho) * Y1);
                    mat U; vec s; mat V;    //declare svd variables
                    svd(U, s, V, Wnew);     //svd of Wnew stored in U, s, and V
                    vec Svec = s;           //SVD is X = U*S*V' where S is diag matrix with s on its diags
                                            //thus no need for line Svec = diag(S) from MatLab

                    //singular value thresholding
                    vec Svecnew = ((1 - a/abs(Svec))%Svec) % (abs(Svec) > a);

                    //update Z
                    mat Znew = U * diagmat(Svecnew) * V.t();
                    Znew = (Znew + Znew.t()) / 2;

                    //Y Update
                    mat Rnew1 = gradD1 + Znew;
                        //Lagrange multiplier update step
                    mat Y1new = Y1 + rho * Rnew1;
                    mat Y2new = Y2 + rho * Rnew2;
                    Y1new = (Y1new + Y1new.t()) / 2;
                    Y2new = (Y2new + Y2new.t()) / 2;

                    //e-values of Xinv
                    vec eigladYnew = real(eig_gen(A.t() * Y1new + Y1new * A + C.t() * (E % Y1new) * C));

                    double logdetLadYnew = sum(log(eigladYnew));
                    double dualYnew = logdetLadYnew - trace(G * Y2new) + m;

                    if (min(eigladYnew) < 0){
                        rho = rho * beta;
                    }
                    else if (dualYnew < dualY +
                            trace(gradD1 * (Y1new - Y1)) +
                            trace(gradD2 * (Y2new - Y2)) -
                            (0.5 / rho) * pow(norm(Y1new - Y1, "fro"), 2) -
                            (0.5 / rho) * pow(norm(Y2new - Y1, "fro"), 2)){
                        rho = rho * beta;
                    }
                    else
                        break;
                }

                /*Quinn's code goes here*/
            }
    }

    /*this is just a place holder as I don't believe CCAMA returns an int but until we figure
    what it actually returns, we'll return a 0 and make the function return an int*/
    return 0;
}

 
