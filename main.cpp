/*******************************************************************************************************/
/*                                                                                                     */
/*                                 CTMRG code for 2D classical models                                  */
/*                                                                                                     */
/*******************************************************************************************************/

/*******************************************************************************************************/
/*                                                                                                     */
/* main.cpp                                                                                            */
/* CTMRG                                                                                               */
/* version 0.1																						   */
/*                                                                                                     */
/* This code is CTMRG algorithm for 2D classical models (Ising, Clock, Potts)                          */
/* Copyright (C) 2016  Jozef Genzor <jozef.genzor@gmail.com>                                           */
/*                                                                                                     */
/*                                                                                                     */
/* This file is part of CTMRG.                                                                         */
/*                                                                                                     */
/* CTMRG is free software: you can redistribute it and/or modify                                       */
/* it under the terms of the GNU General Public License as published by                                */
/* the Free Software Foundation, either version 3 of the License, or                                   */
/* (at your option) any later version.                                                                 */
/*                                                                                                     */
/* CTMRG is distributed in the hope that it will be useful,                                            */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of                                      */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                       */
/* GNU General Public License for more details.                                                        */
/*                                                                                                     */
/* You should have received a copy of the GNU General Public License                                   */
/* along with CTMRG.  If not, see <http://www.gnu.org/licenses/>.                                      */
/*                                                                                                     */
/*******************************************************************************************************/

/*
 * This code is an implementation of the CTMRG algorithm introduced in the articles: 
 *
 * Corner Transfer Matrix Renormalization Group Method
 * J. Phys. Soc. Jpn. 65, pp. 891-894 (1996)
 * http://arxiv.org/abs/cond-mat/9507087v5
 *
 * Corner Transfer Matrix Algorithm for Classical Renormalization Group
 * J. Phys. Soc. Jpn. 66, pp. 3040-3047 (1997) 
 * http://arxiv.org/abs/cond-mat/9705072
 *
 * For linear algebra (matrix diagonalization in particular), 
 * Eigen library (3.2.7) is called. 
 *
 * Compilation under Mac:
 *
 * g++ -m64 -O3 -I/.../Eigen main.cpp -o main.x
 *
 */

/*******************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

#define NDEBUG
#include "SVD"
#include "Eigenvalues"

using namespace Eigen;

const double pi = asin(1.0)*2;

/*******************************************************************************************************/
/*******************************************************************************************************/

void Initialize(const int & q, const int & model, const double & h, const double & g, 
				const double & temperature, double * const W_B, double * const T, double * const C);
MatrixXd Create_DM(const int & q, const int & n, const double * const C);
int Increase(const int & q, const int & m, const int & n);
void Diagonalize_DM(const int & q, const int & n, const int & m, const int & mn, const MatrixXd & DM, double * const E, double * const * const P);
void Extend_and_renormalize_TM(const int & q, const int & n, const int & mn, const double * const T, const double* const W_B,  
							   const double * const * const P, double * const Theta_2, double * const T_new);
void Extend_and_renormalize_CTM(const int & q, const int & n, const int & mn, const double * const T, const double * const C, 
								const double * const * const P, const double * const Theta_2, double * const C_new);
void Normalize_T(const int & q, const int & mn, const double * const T_new, double * const T, std::vector<double> & log_beta);
void Normalize_C(const int & q, const int & mn, const double * const C_new, double * const C, std::vector<double> & log_alpha);
double Calculate_mag(const int & q, const int & model, const int & n, const MatrixXd & DM);
double Calculate_entropy(const int & q, const int & n, const double * const E);
double Trace_C4(const int & q, const int & n, const double * const C);
double Calculate_U(const int & q, const int & model, const int & n, const double * const C, const double * const T); 
double Calculate_F(const int & q, const int & i, const int & n, const double & temperature, const double * const C, 
				   const std::vector<double> & log_beta, const std::vector<double> & log_alpha);
double H1_Ising(const int &, const double & h, const int & s_1, const int & s_2, const int & s_3, const int & s_4);
double H2_Ising(const int &, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4);
double H3_Ising(const int &, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4);
double H1_Clock(const int & q, const double & h, const int & s_1, const int & s_2, const int & s_3, const int & s_4);
double H2_Clock(const int & q, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4);
double H3_Clock(const int & q, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4);
double H1_Potts(const int &, const double & h, const int & s_1, const int & s_2, const int & s_3, const int & s_4);
double H2_Potts(const int &, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4);
double H3_Potts(const int &, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4);

double deltaFunc(const int & i, const int & j);

/*******************************************************************************************************/
/*******************************************************************************************************/

int main () 
{
	clock_t start, end;
	double runTime;
	start = clock(); //time measurement start
    
	int model_r; 
	int q_r;
	double h_r; 
	double initial_r;
	double final_r; 
	double delta_r;
	int m_r;
	int L_r; 
	double g_r; 
	double LIMIT_M_r; 
	double LIMIT_U_r; 
	double LIMIT_F_r; 
	double LIMIT_fF_r; 
	
	FILE* par;
	par = fopen("INIT.txt", "r");
	
	if ( par == NULL ) 
	{ 
		printf("Can't open the input file \"INIT.txt\"\n");
		return 1;
	}
	
	fscanf(par, "%d%*[^\n]", &model_r);
	fscanf(par, "%d%*[^\n]", &q_r);
	fscanf(par, "%lf%*[^\n]", &h_r);
	fscanf(par, "%lf%*[^\n]", &initial_r); 
	fscanf(par, "%lf%*[^\n]", &final_r);  
	fscanf(par, "%lf%*[^\n]", &delta_r);  
	fscanf(par, "%d%*[^\n]", &m_r);		  
	fscanf(par, "%d%*[^\n]", &L_r);
	fscanf(par, "%lf%*[^\n]", &g_r);  
	fscanf(par, "%lf%*[^\n]", &LIMIT_M_r); 
	fscanf(par, "%lf%*[^\n]", &LIMIT_U_r); 
	fscanf(par, "%lf%*[^\n]", &LIMIT_F_r); 	
	fscanf(par, "%lf%*[^\n]", &LIMIT_fF_r); 	
	
	fclose(par);
	
	const int model = model_r; //0 (Ising), 1 (Clock), 2 (Potts)
	if ( model==0 ) // for Ising 
	{
		q_r = 2;
	}
	
	const int q = q_r;  
	const double h = h_r; //1.E-14;
	const double initial = initial_r;
	const double final = final_r;
	const double delta = delta_r;
	const int m = m_r;
	const int L = L_r; //maximal lattice size (i <= L)
	const double g = g_r; //symmetry breaking term (external magnetic field on the boundary)
	const double LIMIT_M = LIMIT_M_r; //precision for magnetization
	const double LIMIT_U = LIMIT_U_r; //precision for inner energy
	const double LIMIT_F = LIMIT_F_r; //precision for free energy - it converges very, very slow
	const double LIMIT_fF = LIMIT_fF_r; //precision for free energy - fast converging 

	if ( model != 0 && model != 1 && model != 2 ) 
	{
		std::cout << "choose correct value for Ising (0), Clock (1) or Potts (2) in INIT.txt" << std::endl;
		abort();
	}
	
	switch (model) 
	{
		case 0:
			printf("# Ising model\n");
			break;
		case 1:
			printf("# %d-states Clock model\n", q);
			break;
		case 2:
			printf("# %d-states Potts model\n", q);
			break;
	}
	printf("# h = %1E\t\tg = %1E\t\tm = %d\t\tL_max = %d\n", h, g, m, L);
	
	FILE* fw;
	if ((fw = fopen("DATA.txt", "w")) == NULL)        //this creates file DATA.txt
	{
		printf("Subor \"DATA.txt\" sa nepodarilo otvorit\n");
		return 1;
	}
	fclose(fw);
	
	fw = fopen("DATA.txt","a");	
	switch (model) 
	{
		case 0:
			fprintf(fw, "# Ising model\n");
			break;
		case 1:
			fprintf(fw, "# %d-states Clock model\n", q);
			break;
		case 2:
			fprintf(fw, "# %d-states Potts model\n", q);
			break;
	}
	
	fprintf(fw, "# m=%d, L_max=%d, h=%1E, g=%1E\n", m, L, h, g);
	fprintf(fw, "# Temp\t\tMagnetization\t\tInternal energy\t\tFree energy\t\tFast free energy\tEntropy\t\t\t\tIter\n");
	fclose(fw);
	
	printf("# Temp\t\tMagnetization\t\tInternal energy\t\tFree energy\t\tFast free energy\tEntropy\t\t\t\tIter\n");
	
	double temperature;
	for (temperature = initial; temperature <= final; temperature += delta) 
	{
		int i = 3;     //actual lattice size (i x i) within each iteration, start with 3x3
		int n = q;     //degrees of freedom counting - we start with n=q
		
		double *W_B;
		double *T;
		double *C;
		
		W_B   =  (double *)malloc(q*q*q*q*sizeof(double));    //Boltzman weight         
		T     =  (double *)malloc(q*q*m*m*sizeof(double));    //transfer matrix (half-row or half-column)
		C     =  (double *)malloc(q*m*m*sizeof(double));      //corner transfer matrix	
		
		Initialize(q, model, h, g, temperature, W_B, T, C);
		
		double entropy = 0;			
		int    numb_of_iter = 0; //number of required iterations
		
		//magnetization
		double mag_new  = 0;
		double mag_prev = 1;
		
		//free energy
		double free_energy_new = 0;
		double free_energy_prev = 1;
		double free_energy_fast_new = 0;
		double free_energy_fast_prev = 1;
				
		std::vector<double> log_beta;  //this vector will store logarithms of normalization coefficients for T
		log_beta.push_back(0);    //T from Initialize() is not normalized, thus first beta=1, and log(beta) = 0
		std::vector<double> log_alpha; //this vector will store logarithms of normalization coefficients for C 
		log_alpha.push_back(0);   //C from Initialize() is not normalized, thus first alpha=1 and log(alpha) = 0 

		//internal energy 
		double U_new  = 0;
		double U_prev = 1;
		
		while (((((fabs(mag_new - mag_prev) > LIMIT_M)) || 
				 ( fabs(U_new - U_prev) > LIMIT_U) || 
				 ( fabs(free_energy_new - free_energy_prev) > LIMIT_F) || 
				 ( fabs(free_energy_fast_new - free_energy_fast_prev) > LIMIT_fF))  
				&& (i <= L) ) || (i<=5) )
		{
			free_energy_prev = free_energy_new;			
			free_energy_new = Calculate_F(q, i, n, temperature, C, log_beta, log_alpha);
			
			free_energy_fast_prev = free_energy_fast_new;
			free_energy_fast_new = - temperature * log_beta[ (i-1)/2 - 1 ];
			
			U_prev = U_new;
			U_new = Calculate_U(q, model, n, C, T);  //U_new is inner energy
			
			MatrixXd* pDM = new MatrixXd(Create_DM(q, n, C));   //density matrix creation - by multiplying four C matrices
			
			mag_prev = mag_new;
			mag_new = Calculate_mag(q, model, n, *pDM); 
			
			int mn = Increase(q, m, n);
			
			double *E;			
			E      =  (double *)malloc(q*n*sizeof(double));       //eigenvalues
			
			double *P[q*n];                                       //projection operator
			for (int j=0; j<q*n; j++) 
			{
				P[j]=(double *)malloc(mn*sizeof(double));
			}
			
			Diagonalize_DM(q, n, m, mn, *pDM, E, P);
			
			delete pDM;
			pDM = NULL;
			
			entropy = Calculate_entropy(q, n, E);
			
			free((void *) E);
			E = NULL;
		
			double *Theta_2;
			Theta_2   =  (double *)malloc(q*q*q*mn*n*sizeof(double)); //auxiliary array
			
			double *T_new;
			T_new     =  (double *)malloc(q*q*mn*mn*sizeof(double));
			
			Extend_and_renormalize_TM(q, n, mn, T, W_B, P, Theta_2, T_new);
			
			double *C_new;
			C_new    =  (double *)malloc(q*mn*mn*sizeof(double)); 
			
			Extend_and_renormalize_CTM(q, n, mn, T, C, P, Theta_2, C_new);
			
			free((void *) Theta_2);   
			Theta_2 = NULL;           
			
			for (int j=0; j<q*n; j++) 
			{
				free(P[j]);       
				P[j]=NULL;	      
			}
			
			Normalize_T(q, mn, T_new, T, log_beta); //normalizing T - dividing by the largest element of array T_new
			
			free((void *) T_new);
			T_new = NULL;
			
			Normalize_C(q, mn, C_new, C, log_alpha); //normalizing C - dividing by the largest element of array C_new
		
			free((void *) C_new);
			C_new = NULL;
			
			n = mn; //increasing n
			
			numb_of_iter = (i-1)/2;
			printf("%1.6E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t\t%d\n", 
				   temperature, mag_new, U_new, free_energy_new, free_energy_fast_new, entropy, numb_of_iter);
			
			i += 2; //lattice size increasing
		}
				
		free((void *) W_B);   
		W_B = NULL;           
		free((void *) T);     
		T = NULL;             
		free((void *) C);     
		C = NULL;  
			
		fw = fopen("DATA.txt","a");	
		fprintf(fw, "%1.6E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t\t%d\n", 
				temperature, mag_new, U_new, free_energy_new, free_energy_fast_new, entropy, numb_of_iter);		
		fclose(fw);
	}
	
	end = clock();        //time measurement end
	runTime = (end - start) / (double) CLOCKS_PER_SEC ;
	printf ("Run time is %f seconds\n", runTime);
	
    return 0;
}

/*******************************************************************************************************/
/*******************************************************************************************************/

void Initialize(const int & q, const int & model, const double & h, const double & g, const double & temperature, 
				double * const W_B, double * const T, double * const C)
{	
	double (* pFunc1)(const int &, const double &, const int &, const int &, const int &, const int &) = NULL;
	double (* pFunc2)(const int &, const double &, const double &, const int &, const int &, const int &, const int &) = NULL;
	double (* pFunc3)(const int &, const double &, const double &, const int &, const int &, const int &, const int &) = NULL;

	switch (model) 
	{
		case 0:
			pFunc1 = &H1_Ising;
			pFunc2 = &H2_Ising;
			pFunc3 = &H3_Ising; 
			break;
		case 1:
			pFunc1 = &H1_Clock;
			pFunc2 = &H2_Clock;
			pFunc3 = &H3_Clock; 
			break;
		case 2:
			pFunc1 = &H1_Potts;
			pFunc2 = &H2_Potts;
			pFunc3 = &H3_Potts; 
			break;
	}
	
	double sum; 
	for (int s_1=0; s_1<q; s_1++) 
	{
		for (int s_2=0; s_2<q; s_2++) 
		{
			for (int s_4=0; s_4<q; s_4++) 
			{
				sum = 0;
				for (int s_3=0; s_3<q; s_3++) 
				{
					W_B[q*q*q*s_1 + q*q*s_2 + q*s_3 + s_4] = exp( - (*pFunc1)(q, h, s_1, s_2, s_3, s_4)/temperature );
					T[q*q*q*s_1 + q*q*s_2 + q*s_3 + s_4]   = exp( - (*pFunc2)(q, h, g, s_1, s_2, s_3, s_4)/temperature );
					sum += exp( - (*pFunc3)(q, h, g, s_1, s_2, s_3, s_4)/temperature );
				}
				C[q*q*s_1 + q*s_2 + s_4] = sum;
			}
		}
	}
}

MatrixXd Create_DM(const int & q, const int & n, const double * const C)
{	
	double sum;
	
	double *AX;                                 
	AX   =  (double *)malloc(q*n*n*sizeof(double)); 
	
	/*  auxiliary array AX initialization  */

	for (int x=0; x<q; x++)
	{
		for (int a=0; a<n; a++)
		{
			for (int c=0; c<n; c++)
			{
				sum = 0;
				for (int b=0; b<n; b++)
				{
					sum += C[n*n*x + n*a + b]*C[n*n*x + n*b + c];
				}
				AX[n*n*x + n*a + c] = sum;
			}
		}
	}
	
	/* normalization of AX */
	
	double norm;
	norm = 0.0;
	for (int j=0; j<q*n*n; j++) 
	{
		norm += AX[j]*AX[j];
	}
	
	norm = sqrt(norm);
	
	for (int j=0; j<q*n*n; j++) 
	{
		AX[j] /= norm;
	}

	MatrixXd DM(q*n, q*n); 
	
	for (int c=0; c<n; c++)
	{
		for (int x=0; x<q; x++)
		{
			for (int d=0; d<n; d++)
			{
				for (int y=0; y<q; y++)
				{
					sum = 0;					
					for (int a=0; a<n; a++)
					{
						sum += AX[n*n*x + n*a + c]*AX[n*n*y + n*a + d];
					}
					DM(q*c + x, q*d + y) = sum;
				}
			}
		}
	}
	
	free((void *) AX);     
	AX = NULL; 
	
	return DM; 
}

void Diagonalize_DM(const int & q, const int & n, const int & m, const int & mn, const MatrixXd & DM, double * const E, double * const * const P)
{	
	/* DM diagonalization */
		
	SelfAdjointEigenSolver<MatrixXd> ES(DM);
	MatrixXd V = ES.eigenvectors();
	//MatrixXd D = ES.eigenvalues().asDiagonal();
	
	for (int i=0; i<(q*n); i++) 
	{
		E[i] = ES.eigenvalues()(i);
	}
	
	if (q*n <= m) 
	{
		for (int j=0; j<(q*n); j++) 
		{
			for (int k=0; k<(q*n); k++) 
			{
				P[j][k] = ES.eigenvectors()(j,k); 
			}
		}
	}	
	else 
	{
		for (int j=0; j<(q*n); j++) 
		{
			for (int k=0; k<mn; k++) 
			{
				P[j][k] = ES.eigenvectors()(j, k + q*n - mn); //this corresponds to the mn largest eigenvalues
			}
		}		
	}
}

void Extend_and_renormalize_TM(const int & q, const int & n, const int & mn, const double * const T, const double* const W_B,  
							   const double * const * const P, double * const Theta_2, double * const T_new)
{	
	double *Theta_1;
	Theta_1   =  (double *)malloc(q*q*mn*n*sizeof(double)); //auxiliary array
	
	double sum;
	
	for (int xi = 0; xi < q; xi++) 
	{
		for (int Xi = 0; Xi < n; Xi++) 
		{
			for (int Sigma_p = 0; Sigma_p < mn; Sigma_p++) 
			{
				for (int sigma = 0; sigma < q; sigma++) 
				{
					sum = 0;
					for (int Sigma = 0; Sigma < n; Sigma++) 
					{
						sum += T[q*n*n*sigma + q*n*Sigma + q*Xi + xi]*P[q*Sigma + sigma][Sigma_p];
					}
					Theta_1[q*mn*n*sigma + q*n*Sigma_p + q*Xi + xi] = sum;
				}
			}
		}
	}
	
	for (int sigma_p = 0; sigma_p < q; sigma_p++) 
	{
		for (int Sigma_p = 0; Sigma_p < mn; Sigma_p++) 
		{
			for (int Xi = 0; Xi < n; Xi++) 
			{
				for (int xi = 0; xi < q; xi++) 
				{
					for (int xi_p = 0; xi_p < q; xi_p++) 
					{
						sum = 0;
						for (int sigma = 0; sigma < q; sigma++) 
						{
							sum += W_B[q*q*q*sigma_p + q*q*sigma + q*xi + xi_p]*Theta_1[q*mn*n*sigma + q*n*Sigma_p + q*Xi + xi];
						}
						Theta_2[q*q*mn*n*sigma_p + q*q*n*Sigma_p + q*q*Xi + q*xi + xi_p] = sum;
					}
				}
			}
		}
	}
	
	free((void *) Theta_1);  
	Theta_1  = NULL;
	
	for (int sigma_p = 0; sigma_p < q; sigma_p++) 
	{
		for (int Sigma_p = 0; Sigma_p < mn; Sigma_p++) 
		{
			for (int Xi_p = 0; Xi_p < mn; Xi_p++) 
			{
				for (int xi_p = 0; xi_p < q; xi_p++) 
				{			
					sum = 0;
					for (int xi = 0; xi < q; xi++) 
					{
						for (int Xi = 0; Xi < n; Xi++) 
						{
							sum += P[q*Xi + xi][Xi_p]*Theta_2[q*q*mn*n*sigma_p + q*q*n*Sigma_p + q*q*Xi + q*xi + xi_p]; 
						}
					}
					T_new[q*mn*mn*sigma_p + q*mn*Sigma_p + q*Xi_p + xi_p] = sum;
				}
			}
		}
	}
}

void Extend_and_renormalize_CTM(const int & q, const int & n, const int & mn, const double * const T, const double * const C, 
								const double * const * const P, const double * const Theta_2, double * const C_new)
{
	double *Theta_3;
	Theta_3  =  (double *)malloc(q*q*q*mn*n*sizeof(double)); //auxiliary array
	
	double sum = 0; 
	
	for (int sigma_p = 0; sigma_p < q; sigma_p++) 
	{
		for (int Sigma_p = 0; Sigma_p < mn; Sigma_p++) 
		{
			for (int Sigma_pp = 0; Sigma_pp < n; Sigma_pp++) 
			{
				for (int xi = 0; xi < q; xi++) 
				{
					for (int xi_p = 0; xi_p < q; xi_p++) 
					{
						sum = 0;
						for (int Xi = 0; Xi < n; Xi++) 
						{
							sum += 
							Theta_2[q*q*mn*n*sigma_p + q*q*n*Sigma_p + q*q*Xi + q*xi + xi_p]
							*C[n*n*xi + n*Xi + Sigma_pp];
						}
						Theta_3[q*q*mn*n*sigma_p + q*q*n*Sigma_p + q*q*Sigma_pp + q*xi + xi_p] = sum;
					}
				}
			}
		}
	}
	
	double *Theta_4;
	Theta_4  =  (double *)malloc(q*q*mn*n*sizeof(double)); //auxiliary array
	
	for (int sigma_p = 0; sigma_p < q; sigma_p++) 
	{
		for (int Sigma_p = 0; Sigma_p < mn; Sigma_p++) 
		{
			for (int Xi_pp = 0; Xi_pp < n; Xi_pp++) 
			{
				for (int xi_p = 0; xi_p < q; xi_p++) 
				{
					sum = 0;
					for (int xi = 0; xi < q; xi++) 
					{
						for (int Sigma_pp = 0; Sigma_pp < n; Sigma_pp++) 
						{
							sum += 
							Theta_3[q*q*mn*n*sigma_p + q*q*n*Sigma_p + q*q*Sigma_pp + q*xi + xi_p]
							*T[q*n*n*xi + q*n*Sigma_pp + q*Xi_pp + xi_p];
						}
					}
					Theta_4[q*mn*n*sigma_p + q*n*Sigma_p + q*Xi_pp + xi_p] = sum;
				}
			}
		}
	}
	
	free((void *) Theta_3);
	Theta_3 = NULL;
	
	for (int sigma_p = 0; sigma_p < q; sigma_p++) 
	{
		for (int Sigma_p = 0; Sigma_p < mn; Sigma_p++) 
		{
			for (int Xi_p = 0; Xi_p < mn; Xi_p++) 
			{
				sum = 0;
				for (int xi_p = 0; xi_p < q; xi_p++) 
				{
					for (int Xi_pp = 0; Xi_pp < n; Xi_pp++) 
					{
						sum += P[q*Xi_pp + xi_p][Xi_p]*Theta_4[q*mn*n*sigma_p + q*n*Sigma_p + q*Xi_pp + xi_p];
					}
				}
				C_new[mn*mn*sigma_p + mn*Sigma_p + Xi_p] = sum;
			}
		}
	}
	
	free((void *) Theta_4);
	Theta_4 = NULL;
}

void Normalize_T(const int & q, const int & mn, const double * const T_new, double * const T, std::vector<double> & log_beta)
{	
	double beta = T_new[0];
	
	for(int j=0; j<q*q*mn*mn; j++) 
	{
		if(beta<T_new[j]) 
		{
			beta = T_new[j];
		}
	}
			
	for(int j=0; j<q*q*mn*mn; j++) 
	{
		T[j] = T_new[j]/beta;
	}	
	
	log_beta.push_back(log(beta));
}

void Normalize_C(const int & q, const int & mn, const double * const C_new, double * const C, std::vector<double> & log_alpha)
{	
	double alpha = C_new[0];
	
	for(int j=0; j<q*mn*mn; j++) 
	{
		if(alpha<C_new[j]) 
		{
			alpha = C_new[j];
		}
	}
	
	for(int j=0; j<q*mn*mn; j++) 
	{
		C[j] = C_new[j]/alpha;
	}
	
	log_alpha.push_back(log(alpha));
}

double Calculate_mag(const int & q, const int & model, const int & n, const MatrixXd & DM)
{
	double mag_new = 0;
	
	switch (model) 
	{
		case 0:
			for (int a=0; a<n; a++) 
			{
				for (int c=0; c<q; c++) 
				{
					mag_new += 2*(c - 0.5)*DM(q*a + c, q*a + c);
				}
			}
			break;
		case 1:
			for (int a=0; a<n; a++) 
			{
				for (int c=0; c<q; c++) 
				{
					mag_new += cos(2*pi*c/q)*DM(q*a + c, q*a + c);
				}
			}
			break;
		case 2:
			for (int a=0; a<n; a++) 
			{
				//mag_new += deltaFunc(c, 0)*DM(q*a + c, q*a + c);
				int c = 0;
				mag_new += DM(q*a + c, q*a + c);
			}
			mag_new = (q*mag_new - 1)/(q-1);
			break;
	}
	
	return mag_new;
}

double Calculate_entropy(const int & q, const int & n, const double * const E)
{
	double entropy = 0;
	
	for (int j=0; j<q*n; j++)						
	{	
		if ( fabs(E[j]) < 1.E-18 )
		{
			continue;
		}
		entropy += - fabs(E[j])*log2(fabs(E[j]));						 
	}
	
	return entropy;
}

double Calculate_U(const int & q, const int & model, const int & n, const double * const C, const double * const T)
{	
	double *Gamma;
	Gamma   =  (double *)malloc(q*q*n*n*sizeof(double)); //auxiliary array for inner energy
	
	double sum;
	
	for (int x=0; x<q; x++) 
	{
		for (int y=0; y<q; y++) 
		{
			for (int Sigma=0; Sigma<n; Sigma++) 
			{
				for (int Xi_pp=0; Xi_pp<n; Xi_pp++) 
				{
					sum = 0;
					for (int Xi=0; Xi<n; Xi++) 
					{
						sum += T[q*n*n*y + q*n*Xi + q*Xi_pp + x]*C[n*n*y + n*Sigma + Xi];
					}
					Gamma[q*n*n*x + n*n*y + n*Sigma + Xi_pp] = sum;
				}
			}
		}
	}
	
	double *Omega;
	Omega   =  (double *)malloc(q*q*n*n*sizeof(double)); //auxiliary array for inner energy
	
	for (int x=0; x<q; x++) 
	{
		for (int y=0; y<q; y++) 
		{
			for (int Sigma=0; Sigma<n; Sigma++) 
			{
				for (int Sigma_pp=0; Sigma_pp<n; Sigma_pp++) 
				{
					sum = 0;
					for (int Xi_pp=0; Xi_pp<n; Xi_pp++) 
					{
						sum += C[n*n*x + n*Xi_pp + Sigma_pp]*Gamma[q*n*n*x + n*n*y + n*Sigma + Xi_pp];
					}
					Omega[q*n*n*x + n*n*y + n*Sigma + Sigma_pp] = sum;
				}
			}
		}
	}
	
	free((void *) Gamma);
	Gamma = NULL;
	
	double *A;
	A   =  (double *)malloc(q*q*sizeof(double)); //auxiliary array for inner energy
	
	
	for (int x=0; x<q; x++) 
	{
		for (int y=0; y<q; y++) 
		{
			sum = 0;
			for (int Sigma=0; Sigma<n; Sigma++) 
			{
				for (int Sigma_pp=0; Sigma_pp<n; Sigma_pp++) 
				{
					sum += Omega[q*n*n*x + n*n*y + n*Sigma + Sigma_pp]*Omega[q*n*n*x + n*n*y + n*Sigma + Sigma_pp];
				}
			}
			A[q*x + y] = sum;
		}
	}
	
	free((void *) Omega);
	Omega = NULL;
	
	double uZ = 0;
	
	for (int x=0; x<q; x++) 
	{
		for (int y=0; y<q; y++) 
		{
			uZ += A[q*x + y];
		}
	}
	
	double U_new = 0;
	
	switch (model) 
	{
		case 0:
			for (int x=0; x<q; x++) 
			{
				for (int y=0; y<q; y++) 
				{
					double z_x = 2*(x - 0.5);
					double z_y = 2*(y - 0.5);
					U_new += z_x*z_y*A[q*x + y];    
				}
			}
			break;
		case 1:
			for (int x=0; x<q; x++) 
			{
				for (int y=0; y<q; y++) 
				{
					double t_x = 2*pi*x/q;
					double t_y = 2*pi*y/q;
					U_new += cos(t_x-t_y)*A[q*x + y];
				}
			}
			break;
		case 2:
			for (int x=0; x<q; x++) 
			{
				for (int y=0; y<q; y++) 
				{
					U_new += deltaFunc(x, y) * A[q*x + y];
					//U_new += deltaFunc(0, y)*A[q*x + y];
					//U_new += deltaFunc(x, 0)*A[q*x + y];
				}
			}
			break;
	}
	
	free((void *) A);
	A = NULL;
	
	U_new *= 2.0;
	//U_new *= q;
	U_new /= - uZ;
	
	return U_new;
	//return ((U_new+1)/(q-1));
}

double Calculate_F(const int & q, const int & i, const int & n, const double & temperature, const double * const C, 
				   const std::vector<double> & log_beta, const std::vector<double> & log_alpha)
{
	double fZ = Trace_C4(q, n, C);  //fZ - is statistical sum for the system of size (i x i)
	long I = i;             //type casting - otherwise i*i might be too big (overflow)
			
	double sum_norm_terms = 0; 
	for (int j=0; j < (i-1)/2; j++) 
	{
		sum_norm_terms += log_alpha[j] + ( (i-1) - 2*(j+1) )*log_beta[j]; 
	}
	
	/* formula for free energy */

	double f = - (4*temperature/(I*I))*(log(fZ)/4 + sum_norm_terms);	
	return f;
}

/*******************************************************************************************************/

double Trace_C4(const int & q, const int & n, const double * const C)
{	
	double *AF;               //auxiliary array
	AF   =  (double *)malloc(q*n*n*sizeof(double));  
	
	/*  auxiliary array AF initialization  */
	
	for (int x=0; x<q; x++)
	{
		for (int a=0; a<n; a++)
		{
			for (int c=0; c<n; c++)
			{
				AF[n*n*x + n*a + c] = 0;
				for (int b=0; b<n; b++)
				{
					AF[n*n*x + n*a + c] += C[n*n*x + n*a + b]*C[n*n*x + n*b + c];
				}
			}
		}
	}
	
	/* normalized statistical sum          */

	double fZ = 0;

	for (int c=0; c<n; c++)
	{
		for (int x=0; x<q; x++)
		{
			for (int a=0; a<n; a++)
			{
				fZ += AF[n*n*x + n*a + c]*AF[n*n*x + n*c + a];
			}
		}
	}
	
	free((void *) AF);     
	AF = NULL;
	
	return fZ;
}

int Increase(const int & q, const int & m, const int & n)
{
	int x;
	if (q*n <= m)
	{
		x = q*n;
	}
	else
	{
		x = m;
	}
	return x;
}
//Hamiltonian in Boltzman weight W_B
double H1_Ising(const int &, const double & h, const int & s_1, const int & s_2, const int & s_3, const int & s_4)              
{
	double z_1 = 2*s_1 - 1;
	double z_2 = 2*s_2 - 1;
	double z_3 = 2*s_3 - 1;
	double z_4 = 2*s_4 - 1;
	
	double h1 = -(z_1*z_2 + z_2*z_3 + z_3*z_4 + z_4*z_1)/2 - h*(z_1 + z_2 + z_3 + z_4)/4;
	return h1;
}
//Hamiltonian in transfer matrix T
double H2_Ising(const int &, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4)    
{
	double z_1 = 2*s_1 - 1;
	double z_2 = 2*s_2 - 1;
	double z_3 = 2*s_3 - 1;
	double z_4 = 2*s_4 - 1;
	
	double h2 = -(z_1*z_2 + 2*z_2*z_3 + z_3*z_4 + z_4*z_1)/2 - h*(z_1 + 2*z_2 + 2*z_3 + z_4)/4 - g*(2*z_2 + 2*z_3)/4;
	return h2;
}
//Hamiltonian in corner transfer matrix C
double H3_Ising(const int &, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4)    
{
	double z_1 = 2*s_1 - 1;
	double z_2 = 2*s_2 - 1;
	double z_3 = 2*s_3 - 1;
	double z_4 = 2*s_4 - 1;
	
	double h3 = -(z_1*z_2 + 2*z_2*z_3 + 2*z_3*z_4 + z_4*z_1)/2 - h*(z_1 + 2*z_2 + 4*z_3 + 2*z_4)/4 - g*(2*z_2 + 4*z_3 + 2*z_4)/4;
	return h3;
}
//Hamiltonian in Boltzman weight W_B
double H1_Clock(const int & q, const double & h, const int & s_1, const int & s_2, const int & s_3, const int & s_4)              
{
	double t_1 = 2*pi*s_1/q;
	double t_2 = 2*pi*s_2/q;
	double t_3 = 2*pi*s_3/q;
	double t_4 = 2*pi*s_4/q;
		
	double h1 = -(cos(t_1 - t_2) + cos(t_2 - t_3) + cos(t_3 - t_4) + cos(t_4 - t_1))/2 
	            - h*(cos(t_1) + cos(t_2) + cos(t_3) + cos(t_4))/4;
	return h1;
}
//Hamiltonian in transfer matrix T
double H2_Clock(const int & q, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4)    
{
	double t_1 = 2*pi*s_1/q;
	double t_2 = 2*pi*s_2/q;
	double t_3 = 2*pi*s_3/q;
	double t_4 = 2*pi*s_4/q;
		
	double h2 = -(cos(t_1-t_2) + 2*cos(t_2-t_3) + cos(t_3-t_4) + cos(t_4-t_1))/2 
	            - h*(cos(t_1) + 2*cos(t_2) + 2*cos(t_3) + cos(t_4))/4
	            - g*(2*cos(t_2) + 2*cos(t_3))/4;
	return h2;
}
//Hamiltonian in corner transfer matrix C
double H3_Clock(const int & q, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4)    
{
	double t_1 = 2*pi*s_1/q;
	double t_2 = 2*pi*s_2/q;
	double t_3 = 2*pi*s_3/q;
	double t_4 = 2*pi*s_4/q;
		
	double h3 = -(cos(t_1-t_2) + 2*cos(t_2-t_3) + 2*cos(t_3-t_4) + cos(t_4-t_1))/2
	            - h*(cos(t_1) + 2*cos(t_2) + 4*cos(t_3) + 2*cos(t_4))/4
	            - g*(2*cos(t_2) + 4*cos(t_3) + 2*cos(t_4))/4;
	return h3;
}
//Hamiltonian in Boltzman weight W_B
double H1_Potts(const int &, const double & h, const int & s_1, const int & s_2, const int & s_3, const int & s_4)              
{	
	double h1 = -(deltaFunc(s_1, s_2) + deltaFunc(s_2, s_3) + deltaFunc(s_3, s_4) + deltaFunc(s_4, s_1))/2 
				- h*(deltaFunc(s_1, 0) + deltaFunc(s_2, 0) + deltaFunc(s_3, 0) + deltaFunc(s_4, 0))/4;
	return h1;
}
//Hamiltonian in transfer matrix T
double H2_Potts(const int &, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4)    
{	
	double h2 = -(deltaFunc(s_1, s_2) + 2*deltaFunc(s_2, s_3) + deltaFunc(s_3, s_4) + deltaFunc(s_4, s_1))/2 
	            - h*(deltaFunc(s_1, 0) + 2*deltaFunc(s_2, 0) + 2*deltaFunc(s_3, 0) + deltaFunc(s_4, 0))/4 
	            - g*(2*deltaFunc(s_2, 0) + 2*deltaFunc(s_3, 0))/4;
	return h2;
}
//Hamiltonian in corner transfer matrix C
double H3_Potts(const int &, const double & h, const double & g, const int & s_1, const int & s_2, const int & s_3, const int & s_4)    
{	
	double h3 = -( deltaFunc(s_1, s_2) + 2*deltaFunc(s_2, s_3) + 2*deltaFunc(s_3, s_4) + deltaFunc(s_4, s_1))/2 
	            - h*(deltaFunc(s_1, 0) + 2*deltaFunc(s_2, 0) + 4*deltaFunc(s_3, 0) + 2*deltaFunc(s_4, 0))/4 
	            - g*(2*deltaFunc(s_2, 0) + 4*deltaFunc(s_3, 0) + 2*deltaFunc(s_4, 0))/4;
	return h3;
}

double deltaFunc(const int & i, const int & j) 
{
	return (i == j);
}
