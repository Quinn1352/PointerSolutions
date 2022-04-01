#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int ccama(A,C,E,G,gamma,n,m,options)

int main()

{
    if (n < 7)
    {
        cout << "At least six input arguments are required" << endl;
    }

    else if (n == 7) {
        int rho = 10;
        double eps_prim = 1.e-5;
        double eps_dual = 1.e-4;
        double maxiter = 1.e5;
        int Xinit = lyap(A, eye(m));
        int Xinit = Xinit;
        mat Zinit(n, m, fill::eye)
        mat Y1init = lyap(A',-Xinit);
        int Y1init = gamma * Y1init / norm(Y1init, 2);
        mat Y2init = eye(n);
        int method = 1;
    }

    else if (n > 8){
            cout << "Too many input arguments." << endl;
}
    // Data Preprocessing 
    
        mat X = Xinit;
        mat Z = Zinit;
        mat Y1 = Y1init;
        mat Y2 = Y2init;

        //AMA implementation

        if method(n == 1) {

            eigLadY = real(eig(A'*Y1 + Y1*A + C' * (E.*Y2) * C));
            logdetLadY = sum(log(eigLadY));
            dualY = logdetLadY - trace(G * Y2) + m;

            // identity matrix
            Ibig = eye(m);
            // backtracking parameters
            beta = 0.5;
            // initial step size
            rho1 = rho0;
            bbfailures = [];

            cout << "rho_bb = " << << endl;
            cout << "rho = " << << endl;
            cout << "eos_prim = " << << endl;
            cout << "res_prim = " << << endl;
            cout << "eps_dual = " << << endl;
            cout << "abs(eta) = " << << endl;
            cout << "iter = " << << endl;
        }
   


}

 