
#include <armadillo>

using namespace arma;

/*
Solves Lyapunov Equation using vectorization
Find X that solve A*X + X*A' = -Q
Input: Matrices A and Q
//assume A to already be stable, and Q to be positive definite 
Output: Matrix X
*/
mat lyap(mat A, mat Q)
{
	mat I(A.n_rows, A.n_cols, fill::eye);

	//Lyap solution using Vectorization
	vec vecQ = vectorise(Q);
	vec vecX = -(solve((kron(I,A) + kron(A,I)), vecQ));

	//Reshape the vector into a matrix
	mat X = reshape(vecX, A.n_rows, A.n_cols);

	//Return X that solves Lyapunov Equation
	return X;
}
