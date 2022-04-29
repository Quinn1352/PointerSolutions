#pragma once

#include <armadillo>

using namespace arma;

struct Output
{
   mat X;      // Covariance matrix resulting from optimization
   mat Z;      // Forcing correlation matrix resulting from optimization
   mat YOne;   // First dual variable resulting from optimization
   mat YTwo;   // Second dual variable resulting from optimization
   double Jp;  // Value of primal function at given step
   double Jd;  // Value of dual function at given step
   double Rp;  // Primal residual at given step
   double dg;     // Duality gap at given step
   int steps;  // Number of steps required to solve problem
   int timer;  // Time taken to solve problem
   bool flag;  // Flags whether the problem was solved before counter reaches maximum
};
