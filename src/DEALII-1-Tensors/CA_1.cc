/* ---------------------------------------------------------------------
 *
 * The first coding assignment to get familiar with tensor calculus related
 * deal.II class templates. This includes:
 * 
 * - Tensor<1,dim>
 * - Tensor<2,dim>
 * - Tensor<4,dim>
 * - Vector<double>
 * - FullMatrix<double>
 * 
 * dim is a template variable which allows to vary e.g. between the two- and
 * three dimensional case. As described in the brief repetition of the 
 * essentials in C++, the respective tensor class templates allow different
 * dimensions as input, i.e. dim=1,dim=2,dim=3 for each respective rank
 *
 * ---------------------------------------------------------------------
 */


#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <iostream>

using namespace dealii;





//----------------------------------------------------

int main ()
{
    /* For now the three dimensional case is considered.
     * This information is essential for deal.II since
     * is suited for arbitrary dimensions due to its
     * template character. Use this variable for all
     * deal.II templates that will be created in the
     * sequent
     * 
     * It is always helpful to read the manual and see
     * how functions are implemented, i.e. what is the
     * return value, how is the function called or what
     * is the input.
     * 
     * Further some functions may be declared 
     * "DEPRECATED" which means they are still usable
     * but will be removed in future releases of the 
     * library -> not recommended to use those
     */
    
    
    const int dim=3;
    //------------------------------------------------
    /* START IMPLEMENTATION HERE                    */
    //------------------------------------------------
    
    
    //------------------------------------------------
    //          EX - 1
    /* Create two tensors of rank one and name them
     * u and v respectively and print them to the screen.
     * Therefore consider the available documentation
     * and manual on the deal.II website
     */
    

    
    //BEGIN - INSERT YOUR CODE HERE
	
    Tensor<1,dim> u({1,2,3});
    Tensor<1,dim> v({2,3,1});	
    
    //END - INSERT YOUR CODE HERE
    
    std::cout<<"The tensor u: "<<u<<std::endl;
    std::cout<<"The tensor v: "<<v<<std::endl;
    //The elements of the tensors are printed row-wise
    //------------------------------------------------
    

    //------------------------------------------------
    //          EX - 2
    /* Compute the scalar product of the vectors
     * u and v, using three different available functions.
     * One of which is called "scalar_prodcut". The others
     * can be found searching in the manual for the tensor
     * class (HINT: check the definitions of the operator "*" and contract)
     */
    
    double result_ex2_method_one = scalar_product(u,v);
    double result_ex2_method_two = u*v;
	double result_ex2_method_three = contract<0,0>(u,v);
    
    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    
    
    std::cout<<"The scalar product of u and v (1st): "<<result_ex2_method_one<<std::endl;
    std::cout<<"The scalar product of u and v(2nd): "<<result_ex2_method_two<<std::endl;    
	std::cout<<"The scalar product of u and v(3rd): "<<result_ex2_method_three<<std::endl;  
    //------------------------------------------------    

    
    //------------------------------------------------
    //          EX - 3
    /* Compute the cross product of the two tensors
     * u and v -> w.
     */
    
    Tensor<1,dim> w = cross_product_3d(u,v);
    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    
    std::cout<<"The cross product of u and v -> w: "<<w<<std::endl;
    //The elements of the tensors are printed row-wise
    //------------------------------------------------         
    
    
    
    //------------------------------------------------
    //          EX - 4
    /* Compute the outer product of the two tensors
     * u and v which renders a second order tensor "A".
     */
    
    Tensor<2,dim> A = outer_product(u,v);
    
    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    
    std::cout<<"The outer product of u and v -> A: "<<A<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------  
    
    //------------------------------------------------
    //          EX - 5
    /* Compute the single contraction of the two tensors
     * A and u = r, i.e. "A dot u", which renders a first order
     * tensor "r". There exist also multiple ways for that
     * computation.
     * Further compute the second order Tensor B as
     * [t_i] =[ 1 5 0] dyadic r
     */
    
    Tensor<1,dim> r;
    Tensor<1,dim> t;
    t[0]=3; t[1]=5; t[2]=9;
    Tensor<2,dim> B;
    
    //BEGIN - INSERT YOUR CODE HERE
    r=contract<1,0>(A,u);
    //Alternative: A*u;
    B=outer_product(t,r);
    //END - INSERT YOUR CODE HERE
    
    std::cout<<"The contraction of A and u -> r: "<<r<<std::endl;
    std::cout<<"The dyadic product t and r -> B: "<<B<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------  
    
    //------------------------------------------------
    //          EX - 6
    /* Compute the transpose of the tensor A and the symmetric
     * part of A
     */
    
    Tensor<2,dim> A_transpose;
    Tensor<2,dim> A_symmetric; //symmetric part of A
    
    //BEGIN - INSERT YOUR CODE HERE
    A_transpose = transpose(A);
    A_symmetric = 0.5 * (A + A_transpose);
    
    //END - INSERT YOUR CODE HERE
    
    std::cout<<"The transpose of A : "<<A_transpose<<std::endl;
    std::cout<<"The symmetric part of A : "<<A_symmetric<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------     
    
    
    //------------------------------------------------
    //          EX - 7
    /* Compute the single contraction of the tensors
     * A and B using the function:
     * contract<index1,index2>(tensor1,tensor2)
     * -> C = A dot B
     * -> D = B dot A
     */
    
    Tensor<2,dim> C;
    Tensor<2,dim> D;
    
    //BEGIN - INSERT YOUR CODE HERE
    C = A * B;
    //Alternative: 
    //C = contract<1,0>(A,B);
    D = contract<1,0>(B,A);
    //END - INSERT YOUR CODE HERE
    std::cout<<"A dot B -> C: "<<C<<std::endl;
    std::cout<<"B dot A -> D: "<<D<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------     
    
    
    //------------------------------------------------
    //          EX - 8
    /* Compute the double contractions of the tensors
     * A and B using the function:
     * double_contract<idx1_t1,idx1_t2,idx2_t1,idx2_t2>(tensor1,tensor2)
     * -> c = A : B
     * -> d = B : A
     */
    
    double c = double_contract<0,0,1,1>(A,B);
    double d = double_contract<0,0,1,1>(B,A);
    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    std::cout<<"A : B -> c: "<<c<<std::endl;
    std::cout<<"B : A -> d: "<<d<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------      
    
    
    //------------------------------------------------
    //          EX - 9
    /* Compute the trace the tensor D = first invariant
     */
    
    double e = trace(D);

    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    std::cout<<"tr(D) : "<<e<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------     
    
    //------------------------------------------------
    //          EX - 10
    /* Compute the Frobenius norm of the tensor D and
     * check the result by computing it "manually" as
     * shown in the exercise (HINT: use std::sqrt(input) to
     * compute the square root of a number)
     */
    
    double f = D.norm(); //using deal.II function
    double g = std::sqrt(double_contract<0,0,1,1>(D,D)); //manually
        
    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    std::cout<<"||D|| (Frobenius) deal.II fun: "<<f<<std::endl;
    std::cout<<"||D|| (Frobenius) manually: "<<g<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------    


    //------------------------------------------------
    //          EX - 11
    /* Compute the 2nd Invariant of the tensor D
     */
    
    double scnd_inv_D = 0.5* (trace(D)*trace(D) - trace(D*D));
    
    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    std::cout<<"The second invariant of D: "<<scnd_inv_D<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------         
    
    //------------------------------------------------
    //          EX - 12
    /* Compute the 3rd Invariant of the tensor D
     */
    
    double thrd_inv_D = determinant(D);
        
    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    std::cout<<"The third invariant of D: "<<thrd_inv_D<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------        
    
    //------------------------------------------------
    //          EX - 13
    /* Compute the 4th-order tensor C_fthrd as:
     * C dyadic D
     */
    
    Tensor<4,dim> C_fthrd = outer_product(C,D); 
        
    //BEGIN - INSERT YOUR CODE HERE

    
    //END - INSERT YOUR CODE HERE
    std::cout<<"4th-order tensor C dyadic D -> C_fthrd: "<<C_fthrd<<std::endl;
    //This prints the tensor row-wise to the console
    //------------------------------------------------   
    
    //------------------------------------------------
    //          EX - 14
    /* Initialise the vectors "vec_u" and "vec_v" as:
     * vec_u = [3 5 1]  vec_v = [1 2 4]
     * Further initialise the matrices mat_A and mat_B:
     * mat_A:   [1 4 3
     *           5 7 9
     *           8 9 -2]
     * mab_B:   [2 5 3
     *           -3 14 99
     *           23 17 19]
     * 
     * Compute the following:
     * - the scalar prodcut vo vec_u and vec_v -> vec_u_dot_vec_v
     * - the matrix-vector product mat_A * vec_v
     * - the matrix-vector product mat_A' * vec_v (transpose of mat_A)
     * - the matrix-matrix product mat_A * mat_B
     * - the matrix-matrix product mat_A' * mat_B
     * - the matrix-matrix product mat_A * mat_B'
     * 
     * HINT: see the documentation for the class templates
     * "Vector<double> and FullMatrix<double> in the deal.II 
     * documentation!!
     */
    std::cout<<"\n\n\tVector - Matrix Examples \n\n"<<std::endl;
    FullMatrix<double> mat_A(3,3); 
    FullMatrix<double> mat_B(3,3); 
    FullMatrix<double> mat_A_times_mat_B(3,3); 
    FullMatrix<double> mat_Atransposed_times_mat_B(3,3); 
    FullMatrix<double> mat_A_times_mat_Btransposed(3,3); 
    Vector<double> vec_u(3);
    Vector<double> vec_v(3);
    double vec_u_dot_vec_v;
    Vector<double> mat_A_dot_vec_v(3);
    Vector<double> mat_Atransposed_dot_vec_v(3);
    
        
    //BEGIN - INSERT YOUR CODE HERE
    vec_u[0]=3; vec_u[1]=5; vec_u[2]=1;
    vec_v[0]=1; vec_v[1]=2; vec_v[2]=4;
    
    vec_u_dot_vec_v = vec_u*vec_v;
    
    //Opt 1 (,)
    mat_A(0,0) = 1;
    //Opt 2 [][]
    mat_A[0][1] = 4; mat_A[0][2] = 3;
    mat_A[1][0] = 5; mat_A[1][1] = 7; mat_A[1][2] = 9;
    mat_A[2][0] = 8; mat_A[2][1] = 9; mat_A[2][2] = -2;
    
    mat_B[0][0] = 2; mat_B[0][1] = 5; mat_B[0][2] = 3;
    mat_B[1][0] = -3; mat_B[1][1] = 14; mat_B[1][2] = 99;
    mat_B[2][0] = 23; mat_B[2][1] = 17; mat_B[2][2] = 19;    
    
	mat_A.vmult(mat_A_dot_vec_v,vec_v);
	mat_A.Tvmult(mat_Atransposed_dot_vec_v,vec_v);
    mat_A.mmult(mat_A_times_mat_B,mat_B);
    mat_A.Tmmult(mat_Atransposed_times_mat_B,mat_B);
    mat_A.mTmult(mat_A_times_mat_Btransposed,mat_B);    

    
    //END - INSERT YOUR CODE HERE
    
    //matrices can not be print with "<<" directly -> use the function *.print(std::cout)
    std::cout<<"The vector u: "<<vec_u<<std::endl; 
    std::cout<<"The vector v: "<<vec_v<<std::endl;
    std::cout<<"The matrix A: "<<std::endl; mat_A.print(std::cout);
    std::cout<<"The matrix B: "<<std::endl; mat_B.print(std::cout);
	std::cout<<"The scalar product of u and v: "<<vec_u_dot_vec_v<<std::endl;
	std::cout<<"The matrix-vector product of A and v: "<<mat_A_dot_vec_v<<std::endl;
	std::cout<<"The matrix-vector product of A' and v: "<<mat_Atransposed_dot_vec_v<<std::endl;
    std::cout<<"The matrix product A times B: "<<std::endl; mat_A_times_mat_B.print(std::cout,10,5);
    std::cout<<"The matrix product A' times B: "<<std::endl; mat_Atransposed_times_mat_B.print(std::cout,10,5);
    std::cout<<"The matrix product A times B': "<<std::endl; mat_A_times_mat_Btransposed.print(std::cout,10,5);
    
    
    //This prints the tensor row-wise to the console
    //------------------------------------------------     
    
}







//----------------------------------------------------