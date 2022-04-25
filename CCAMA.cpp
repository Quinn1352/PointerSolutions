#include <iostream>
#include <armadillo>
#include <math.h>
#include "Lyap.h"
#include "Options.h"
#include "Output.h"

using namespace std;
using namespace arma;


Output ccama (mat A, mat C, mat E, mat G, int gamma, int n, int m, Options options) 
{       
    //declarations needed for output
    Output output; 
    mat Y1new;
    mat Y2new;
    mat Xnew;
    mat Znew;

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
            vector<int> bbfailures;   //empty 0x0 matrix; not sure the use; come back to

            /*print function to print all 
              matrix values needs to go hear*/

            /*Timer should go here*/


            //Use AMA to solve the gamma parametized problem
            int AMAstep;
            for ( AMAstep = 1; AMAstep <= MaxIter; AMAstep++) {

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
                double dualYnew;

                for (int j = 1; j <= 50; j++) {

                    int a = gamma / rho;

                    //Z minimization step
                    mat Wnew = -(A * Xnew + Xnew * A.t() + (1 / rho) * Y1);
                    mat U; vec s; mat V;    //declare svd variables
                    svd(U, s, V, Wnew);     //svd of Wnew stored in U, s, and V
                    vec Svec = s;           //SVD is X = U*S*V' where S is diag matrix with s on its diags
                                            //thus no need for line Svec = diag(S) from MatLab

                    //singular value thresholding
                    Svecnew = ((1 - a/abs(Svec))%Svec) % (abs(Svec) > a);

                    //update Z
                    Znew = U * diagmat(Svecnew) * V.t();
                    Znew = (Znew + Znew.t()) / 2;

                    //Y Update
                    mat Rnew1 = gradD1 + Znew;
                        //Lagrange multiplier update step
                    Y1new = Y1 + rho * Rnew1;
                    Y2new = Y2 + rho * Rnew2;
                    Y1new = (Y1new + Y1new.t()) / 2;
                    Y2new = (Y2new + Y2new.t()) / 2;

                    //e-values of Xinv
                    vec eigladYnew = real(eig_gen(A.t() * Y1new + Y1new * A + C.t() * (E % Y1new) * C));

                    double logdetLadYnew = sum(log(eigladYnew));
                    dualYnew = logdetLadYnew - trace(G * Y2new) + m;

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
                //primal residual
                mat Rnew1 = (A * Xnew) + (Xnew * A.t()) + Znew;
                mat Rnew2 = E % (C * Xnew * C.t()) - G;
                double res_prim = sqrt(pow(norm(Rnew1, 'fro'), 2) + pow(norm(Rnew2, 'fro'), 2));

                //calculating the duality gap
                //doesn't like log_det(X) because trying to store in a double
                cx_double eta =   -log_det(X) + (gamma * sum(Svecnew) - dualYnew); //not sure what the dualYnew is supposed to be

                if (AMAstep % 100 == 0) {
                    cout << rho1 << "\t" << rho << "\t" << eps_prim << "\t" << res_prim << "\t" <<
                        eps_dual << "\t" << abs(eta) << "\t" << AMAstep << endl;
                }

                //BB stepsize selection starts here
                mat Xnew1 = (A.t() * Y1new) + (Y1new * A) + (C.t() * (E % Y2new) * C);
                Xnew1 = solve(Xnew1, Ibig);
                Xnew1 = (Xnew1 + Xnew1.t()) / 2;

                //gradient of dual function
                mat gradD1new = (A * Xnew1) + (Xnew1 * A.t());
                mat gradD2new = (E % (C * Xnew1 * C.t())) - G;
                rho1 = real(pow(norm(Y1new - Y1, 'fro'), 2) + pow(norm(Y2new - Y2, 'fro'), 2) /
                    (trace((Y1new - Y1) * (gradD1 - gradD1new)) + trace((Y2new - Y2) * (gradD2 - gradD2new))));
                mat RHO1(1, 1);
                RHO1(1, 1) = rho1;
                if (rho1 < 0 || RHO1.has_nan()) {
                    rho1 = rho0;
                    bbfailures.push_back(AMAstep);  
                }
                //end bb stepsize selection

                //record path of relevant primal and dual quantities

                

                output.Jp = -log(det(Xnew)) + (gamma * sum(svd(Znew)));
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
                }

                Y1 = Y1new;
                Y2 = Y2new;
                dualY = dualYnew;  
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

    /*this is just a place holder as I don't believe CCAMA returns an int but until we figure
    what it actually returns, we'll return a 0 and make the function return an int*/
    return output;
}

 
