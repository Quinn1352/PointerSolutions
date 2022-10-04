#include <iostream>
#include <iomanip>
#include <armadillo>
#include <math.h>
#include "Lyap.h"
#include "Options.h"
#include "Output.h"
#include "CCAMA.h"
#include <chrono>

using namespace std;
using namespace arma;
using namespace chrono;


int main() {

	//number of masses
	int N = 20;

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
	mat A(2*N, 2*N);
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

	options.useOptions = true;

	
	auto tic = high_resolution_clock::now();
	Output output = ccama(A, C, E, G, gamma, n, m, options);
	auto toc = high_resolution_clock::now();

	auto duration = duration_cast<seconds>(toc - tic);
	cout << "CCAMA exection time was " << duration.count() << " seconds" << endl;



}

//*/