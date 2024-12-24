/* ---------------------------------------------------------------------
 *
 * Implementation of tensor calculus-related mathematical operations
 * using deal.II class templates. This includes:
 * 
 * - Tensor<1,dim>
 * - Tensor<2,dim>
 * - Tensor<4,dim>
 * - Vector<double>
 * - FullMatrix<double>
 * 
 * Here, dim is a template variable that allows switching between
 * different dimensions, e.g., dim=2 or dim=3 for specific ranks.
 * The computations involve mathematical operations such as scalar
 * products, cross products, outer products, contractions, invariants,
 * and norms.
 * 
 * ---------------------------------------------------------------------
 */

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <iostream>

using namespace dealii;

int main()
{
    // Setting the dimensionality of the tensors and matrices.
    const int dim = 3;

    // ------------------------------------------------
    // Step 1: Initialize and print two rank-1 tensors u and v
    Tensor<1, dim> u({1, 2, 3});
    Tensor<1, dim> v({2, 3, 1});

    // Displaying the initialized tensors
    std::cout << "The tensor u: " << u << std::endl;
    std::cout << "The tensor v: " << v << std::endl;

    // ------------------------------------------------
    // Step 2: Compute the scalar product of u and v using various methods.
    double result_ex2_method_one = scalar_product(u, v);
    double result_ex2_method_two = u * v;
    double result_ex2_method_three = contract<0, 0>(u, v);

    // Displaying the results of the scalar products
    std::cout << "The scalar product of u and v (1st): " << result_ex2_method_one << std::endl;
    std::cout << "The scalar product of u and v (2nd): " << result_ex2_method_two << std::endl;
    std::cout << "The scalar product of u and v (3rd): " << result_ex2_method_three << std::endl;

    // ------------------------------------------------
    // Step 3: Compute the cross product of u and v to get w.
    Tensor<1, dim> w = cross_product_3d(u, v);

    // Displaying the cross product
    std::cout << "The cross product of u and v -> w: " << w << std::endl;

    // ------------------------------------------------
    // Step 4: Compute the outer product of u and v to obtain a rank-2 tensor A.
    Tensor<2, dim> A = outer_product(u, v);

    // Displaying the outer product result
    std::cout << "The outer product of u and v -> A: " << A << std::endl;

    // ------------------------------------------------
    // Step 5: Perform a single contraction of A and u to compute r, and construct B.
    Tensor<1, dim> r = contract<1, 0>(A, u);
    Tensor<1, dim> t;
    t[0] = 3; t[1] = 5; t[2] = 9;
    Tensor<2, dim> B = outer_product(t, r);

    // Displaying the results of the contraction and dyadic product
    std::cout << "The contraction of A and u -> r: " << r << std::endl;
    std::cout << "The dyadic product of t and r -> B: " << B << std::endl;

    // ------------------------------------------------
    // Step 6: Compute the transpose and symmetric part of A.
    Tensor<2, dim> A_transpose = transpose(A);
    Tensor<2, dim> A_symmetric = 0.5 * (A + A_transpose);

    // Displaying the transpose and symmetric part
    std::cout << "The transpose of A: " << A_transpose << std::endl;
    std::cout << "The symmetric part of A: " << A_symmetric << std::endl;

    // ------------------------------------------------
    // Step 7: Perform single contractions of A and B to compute tensors C and D.
    Tensor<2, dim> C = contract<1, 0>(A, B);
    Tensor<2, dim> D = contract<1, 0>(B, A);

    // Displaying the results of single contractions
    std::cout << "A dot B -> C: " << C << std::endl;
    std::cout << "B dot A -> D: " << D << std::endl;

    // ------------------------------------------------
    // Step 8: Compute the double contractions of A and B.
    double c = double_contract<0, 0, 1, 1>(A, B);
    double d = double_contract<0, 0, 1, 1>(B, A);

    // Displaying the results of double contractions
    std::cout << "A : B -> c: " << c << std::endl;
    std::cout << "B : A -> d: " << d << std::endl;

    // ------------------------------------------------
    // Step 9: Compute the trace of tensor D.
    double e = trace(D);

    // Displaying the trace
    std::cout << "tr(D): " << e << std::endl;

    // ------------------------------------------------
    // Step 10: Compute the Frobenius norm of D both directly and manually.
    double f = D.norm();
    double g = std::sqrt(double_contract<0, 0, 1, 1>(D, D));

    // Displaying the Frobenius norm
    std::cout << "||D|| (Frobenius, deal.II): " << f << std::endl;
    std::cout << "||D|| (Frobenius, manual): " << g << std::endl;

    // ------------------------------------------------
    // Step 11: Compute the 2nd invariant of D.
    double scnd_inv_D = 0.5 * (trace(D) * trace(D) - trace(D * D));

    // Displaying the second invariant
    std::cout << "The second invariant of D: " << scnd_inv_D << std::endl;

    // ------------------------------------------------
    // Step 12: Compute the 3rd invariant (determinant) of D.
    double thrd_inv_D = determinant(D);

    // Displaying the third invariant
    std::cout << "The third invariant of D: " << thrd_inv_D << std::endl;

    // ------------------------------------------------
    // Step 13: Construct a fourth-order tensor C_fthrd as C dyadic D.
    Tensor<4, dim> C_fthrd = outer_product(C, D);

    // Displaying the fourth-order tensor
    std::cout << "4th-order tensor C dyadic D -> C_fthrd: " << C_fthrd << std::endl;

    // ------------------------------------------------
    // Step 14: Perform vector and matrix operations.
    FullMatrix<double> mat_A(3, 3);
    FullMatrix<double> mat_B(3, 3);
    Vector<double> vec_u(3);
    Vector<double> vec_v(3);

    vec_u[0] = 3; vec_u[1] = 5; vec_u[2] = 1;
    vec_v[0] = 1; vec_v[1] = 2; vec_v[2] = 4;
    double vec_u_dot_vec_v = vec_u * vec_v;

    mat_A[0][0] = 1; mat_A[0][1] = 4; mat_A[0][2] = 3;
    mat_A[1][0] = 5; mat_A[1][1] = 7; mat_A[1][2] = 9;
    mat_A[2][0] = 8; mat_A[2][1] = 9; mat_A[2][2] = -2;

    mat_B[0][0] = 2; mat_B[0][1] = 5; mat_B[0][2] = 3;
    mat_B[1][0] = -3; mat_B[1][1] = 14; mat_B[1][2] = 99;
    mat_B[2][0] = 23; mat_B[2][1] = 17; mat_B[2][2] = 19;

    Vector<double> mat_A_dot_vec_v(3);
    Vector<double> mat_Atransposed_dot_vec_v(3);
    FullMatrix<double> mat_A_times_mat_B(3, 3);
    FullMatrix<double> mat_Atransposed_times_mat_B(3, 3);
    FullMatrix<double> mat_A_times_mat_Btransposed(3, 3);

    mat_A.vmult(mat_A_dot_vec_v, vec_v);
    mat_A.Tvmult(mat_Atransposed_dot_vec_v, vec_v);
    mat_A.mmult(mat_A_times_mat_B, mat_B);
    mat_A.Tmmult(mat_Atransposed_times_mat_B, mat_B);
    mat_A.mTmult(mat_A_times_mat_Btransposed, mat_B);

    // Displaying results of vector and matrix operations
    std::cout << "The scalar product of u and v: " << vec_u_dot_vec_v << std::endl;
    std::cout << "The matrix-vector product of A and v: " << mat_A_dot_vec_v << std::endl;
    std::cout << "The matrix-vector product of A' and v: " << mat_Atransposed_dot_vec_v << std::endl;
    std::cout << "The matrix product A times B: " << std::endl; mat_A_times_mat_B.print(std::cout);
    std::cout << "The matrix product A' times B: " << std::endl; mat_Atransposed_times_mat_B.print(std::cout);
    std::cout << "The matrix product A times B': " << std::endl; mat_A_times_mat_Btransposed.print(std::cout);

    return 0;
}
