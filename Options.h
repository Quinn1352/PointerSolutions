#pragma once
#include <armadillo>

using namespace arma;

struct Options
{
   double rho;          // Initial step size
   double epsPrim;   // Primal constraint tolerance
   double epsDual;   // Duality gap tolerance
   int maxIter;      // Maximum number of iterations
   mat xInit;        // Initial value of matrix X
   mat zInit;        // Initial value of matrix Z
   mat yOneInit;     // Initial value of dual matrix Y1
   mat yTwoInit;     // Initial value of dual matrix Y2
   int method;       // Method selection option (Most likely unused)
   bool useOptions; //used for CCAMA to determine whether to use options or to use default inputs
};
