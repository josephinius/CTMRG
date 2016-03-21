# CTMRG

This code is an implementation of the CTMRG algorithm introduced in the articles: 

Corner Transfer Matrix Renormalization Group Method
J. Phys. Soc. Jpn. 65, pp. 891-894 (1996)
http://arxiv.org/abs/cond-mat/9507087v5

Corner Transfer Matrix Algorithm for Classical Renormalization Group
J. Phys. Soc. Jpn. 66, pp. 3040-3047 (1997) 
http://arxiv.org/abs/cond-mat/9705072

For linear algebra (matrix diagonalization in particular), 
Eigen library (3.2.7) is called. 

Compilation under Mac:

g++ -m64 -O3 -I/.../Eigen main.cpp -o main.x
