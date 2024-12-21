
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>


#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>

#include "HyperCubeWithRefinedHole.h"
#include "StrainMeasures.h"
#include "NeoHookeanMaterial.h"


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

using namespace dealii;

/*! \brief Coding Assignment 4
 *  \author Dominic Soldner
 * 	\version 3.0
 * \date Jan 2020
 *  
*/

template <int dim>
class Solid
{
public:
	Solid(unsigned int load_steps, unsigned int poly_degree, double load_magnitude,
					double mu, double lambda);

	virtual ~Solid(	);

	void run();

private:
	
	/*!
	 * Generate a mesh using function from namespace HyperCubeWithRefinedHole
	 */
	void make_grid();
	/*!
	 * Distribute dof's based on a given Finite Element space and allocating memory for the
	 * sparse matrix and all used vectors.
	 */
	void system_setup();
	/*!Assemble the linear system for the elasticity problem*/
	void assemble_system();
	/*!Set hanging node and Dirichlet constraints*/
	void make_constraints(const int &it_nr);
	/*!Newton-Raphson algorithm looping over all newton iterations*/
	void solve_load_step_NR(Vector<double> &solution_delta);
	/*!Solve the linear system as assemble via assemble_system()*/
	std::pair<unsigned int, double> solve_linear_system(Vector<double> &newton_update);
	
	Vector<double> get_total_solution(const Vector<double> &solution_delta) const;

	void output_results() const;

	Triangulation<dim>               triangulation;

	const unsigned int               degree;
	const FESystem<dim>              fe;
	DoFHandler<dim>                  dof_handler_ref;
	const unsigned int               dofs_per_cell;
	const FEValuesExtractors::Vector u_fe;
	
	enum
	{
		u_dof = 0
	};

	const QGauss<dim>                qf_cell;
	const QGauss<dim - 1>            qf_face;
	const unsigned int               n_q_points;
	const unsigned int               n_q_points_f;


	AffineConstraints<double>                 constraints;

	SparsityPattern             sparsity_pattern;
	SparseMatrix<double>        tangent_matrix;
	Vector<double>              system_rhs;
	Vector<double>              solution_n;
	Vector<double>				solution_delta;

	double mu;
	double lambda;
	double load_magnitude;
	unsigned int load_steps;
	unsigned int current_load_step=0;
	unsigned int max_number_newton_iterations=10;
	unsigned int multiplier_max_iterations_linear_solver=1;
	double error_tolerance_displacement=1e-6;
	double error_tolerance_residual=1e-6;
	unsigned int id_Dirichlet_boundary = 5;
	unsigned int id_Neumann_boundary = 6;	
	unsigned int nbr_adaptive_refinements = 2;
	//-------------------------------------------------------------------------
	/*!A struct used to keep track of data needed as convergence criteria. As typical for a struct all member functions and variables are public
	 */
	struct Errors
	{
		Errors()
		:
		u(1.0)
		{}

		void reset()
		{
			u = 1.0;
		}
		void normalise(const Errors &err)
		{
			if (err.u != 0.0)
			{
				u /= err.u;
			}
		}
		//member variables
		double u;
	};
	/*Multiple instances of the struct Errors for normalisation
	the current error, the initial error and the normalised error
	for the residual but also used for the solution delta
	*/
	Errors error_residual, error_residual_0, error_residual_norm;
			
		/*!Compute the 2-norm of the residual vector
	 */
	void get_error_residual(Errors &error_residual);

	/*!Output to the console
	 */
	void print_conv_header();
	/*!Output to the console
	 */
	void print_conv_footer();
	//-------------------------------------------------------------------------				
};




template <int dim>
Solid<dim>::Solid(unsigned int load_steps, unsigned int poly_degree, double load_magnitude,
				double mu, double lambda)
:
degree(poly_degree),
fe(FE_Q<dim>(degree), dim), // displacement
dof_handler_ref(triangulation),
dofs_per_cell (fe.dofs_per_cell),
u_fe(0),
qf_cell(2),
qf_face(2),
n_q_points (qf_cell.size()),
n_q_points_f (qf_face.size()),
mu(mu),
lambda(lambda),
load_magnitude(load_magnitude),
load_steps(load_steps)
{
}


template <int dim>
Solid<dim>::~Solid()
{
	dof_handler_ref.clear();
}


template <int dim>
void Solid<dim>::run()
{
	make_grid();
	system_setup();
	//output initial values (here: =0)
	output_results();
	//Loop over the number of load_steps (see class declaration)
	for (current_load_step=1; current_load_step <= load_steps; current_load_step++)
	{
			/*Always reset the increment vector - not to be mistaken with
			 the newton update!!!*/
			solution_delta = 0.0;
			/*Compute for the current load step the incremental solution using
			 Newton-Rapshon*/
			solve_load_step_NR(solution_delta);
			/*add the converged delta to the solution - not to be mistaken with
			 the newton update!!!*/
			solution_n += solution_delta;
			output_results();
	}
}




template <int dim>
void Solid<dim>::make_grid()
{
	HyperCubeWithRefinedHole::generate_grid<dim>(triangulation,
												 nbr_adaptive_refinements,
												id_Dirichlet_boundary,
												id_Neumann_boundary);	  
	
	std::ofstream out_ucd("Grid_HyperCubeWithRefinedHole.inp");
	GridOut grid_out;
	GridOutFlags::Ucd ucd_flags(true,true,true);
	grid_out.set_flags(ucd_flags);
	grid_out.write_ucd(triangulation, out_ucd);
	std::cout<<"Mesh written to Grid_HyperCubeWithRefinedHole.inp "<<std::endl;
}





template <int dim>
void Solid<dim>::system_setup()
{

	
	dof_handler_ref.distribute_dofs(fe);


 	DoFRenumbering::Cuthill_McKee(dof_handler_ref);
// 	DoFRenumbering::random(dof_handler_ref);

	
	constraints.clear();
	DoFTools::make_hanging_node_constraints (dof_handler_ref,constraints);
	constraints.close();
	
	std::cout << "Triangulation:"
				<< "\n\t Number of active cells: " << triangulation.n_active_cells()
				<< "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs()
				<< std::endl;


	tangent_matrix.clear();
	const types::global_dof_index n_dofs_u = dof_handler_ref.n_dofs();

	/*Due to internal data structure of deal.ii classes (estimation of memory) a DynamicSparsityPattern is used
	 * first (different structre than the SparsityPattern itself) - Details in the Sparsity pattern module
	 */
	DynamicSparsityPattern dsp(n_dofs_u, n_dofs_u);
	DoFTools::make_sparsity_pattern(dof_handler_ref,
								dsp,
								constraints,
								true);//true);//dont keep constraint dof sparsity pattern entries
	sparsity_pattern.copy_from (dsp);

	unsigned int number_entries = sparsity_pattern.n_nonzero_elements();
	std::cout<<"Size of sparsity-pattern: "<<number_entries<<std::endl;
	std::ofstream out ("sparsity_pattern1.svg");
	sparsity_pattern.print_svg (out);	
	
	tangent_matrix.reinit (sparsity_pattern);
	system_rhs.reinit(n_dofs_u);
	solution_delta.reinit(n_dofs_u);
	solution_n.reinit(n_dofs_u);
}


template <int dim>
void Solid<dim>::solve_load_step_NR(Vector<double> &solution_delta)
{
	/*A vector used for all the newton increments*/
	Vector<double> newton_update(dof_handler_ref.n_dofs());
	/*Reset the structs used for convergence criteria*/
	error_residual.reset();
	error_residual_0.reset();
	error_residual_norm.reset();
	/*Print info to the screen*/
	print_conv_header();

	unsigned int newton_iteration = 0;
	for (; newton_iteration <= max_number_newton_iterations;
			++newton_iteration)
	{
		std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;


		//BEGIN - INSERT YOUR CODE HERE
		//RESET THE TANGENT MATRIX, THE RHS
		//CALL THE FUNCTIONS make_constraints (WITH THE CORRECT PARAMETER)
		//AND ASSEMBLE_SYSTEM
		tangent_matrix = 0.0;
		system_rhs = 0.0;
		make_constraints(newton_iteration);
		assemble_system();
		
		//END - INSERT YOUR CODE HERE

		get_error_residual(error_residual);
		if (newton_iteration == 0)
		{
			error_residual_0 = error_residual;
		}
		error_residual_norm = error_residual;
		error_residual_norm.normalise(error_residual_0);

		/*Problem has to be solved at least once*/
		if (newton_iteration > 0 && error_residual_norm.u <= error_tolerance_residual)
		{
			std::cout << " CONVERGED! " << std::endl;
			/*Print info to the screen*/
			print_conv_footer();
			break;
		}
		
		
		const std::pair<unsigned int, double>
		lin_solver_output = solve_linear_system(newton_update);
		//BEGIN - INSERT YOUR CODE HERE
		//ADD THE NEWTION INCREMENT TO THE LOAD STEP DELTA solution_delta
		solution_delta += newton_update;
		//END - INSERT YOUR CODE HERE
		
		
		/*Print info to the screen*/
		std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
					<< std::scientific << lin_solver_output.first << "  "
					<< lin_solver_output.second << "  " << error_residual_norm.u 
					<< "  " << std::endl;
	}
	  AssertThrow (newton_iteration < max_number_newton_iterations,
               ExcMessage("No convergence in nonlinear solver!"));	
}

template <int dim>
void Solid<dim>::print_conv_header()
{
	double step_fraction = double(current_load_step)/double(load_steps);
	double current_load = load_magnitude * step_fraction;
	std::cout << "\nStep " << current_load_step << " out of "
			<< load_steps << " load steps with current load: "<<step_fraction<<"*"
			<<load_magnitude<<" = "<<current_load << std::endl;

	const unsigned int l_width = 90;
	for (unsigned int i = 0; i < l_width; ++i)
	{
		std::cout << "_";
	}
	std::cout << std::endl;

	std::cout << "           SOLVER STEP            "
				<< " |  LIN_IT   LIN_RES    RES_NORM    "
				<< std::endl;

	for (unsigned int i = 0; i < l_width; ++i)
	{
		std::cout << "_";
	}
	std::cout << std::endl;
}

template <int dim>
void Solid<dim>::print_conv_footer()
{
	const unsigned int l_width = 90;
	for (unsigned int i = 0; i < l_width; ++i)
	{
		std::cout << "_";
	}
	std::cout << std::endl;

	std::cout << "Errors:" << std::endl
				<< "Rhs: \t\t" << error_residual.u << std::endl
				<< std::endl;
}


template <int dim>
void Solid<dim>::get_error_residual(Errors &error_residual)
{
	/*This step is necessary if the entry of the vector
	 at a constrained dof is not zero - this depends on 
	 the way constraints are imposed; To be sure it is 
	 safer to only consider the unconstrained entries anyway*/
	Vector<double> error_res(dof_handler_ref.n_dofs());
	for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
	{
		if (!constraints.is_constrained(i))
		{
			error_res(i) = system_rhs(i);
		}
	}
	error_residual.u = error_res.l2_norm();
}


template <int dim>
Vector<double>
Solid<dim>::get_total_solution(const Vector<double> &solution_delta) const
{
	Vector<double> solution_total(solution_n);
	solution_total += solution_delta;
	return solution_total;
}


template <int dim>
void Solid<dim>::make_constraints(const int &it_nr)
{
    std::cout << " CST " << std::flush;
	/*Check the current number of newton iterations and decide based on 
	 the discussion in the exercise after how many iterations you could 
	 leave this function*/
	//BEGIN - INSERT YOUR CODE HERE
    if (it_nr >= 1)
	{
		return;
	}
	//END - INSERT YOUR CODE HERE

	/*Clear and write hanging node and Dirichlet constraints into
	 the affine_constraints constraints*/
    constraints.clear();
	DoFTools::make_hanging_node_constraints (dof_handler_ref,constraints);
    const bool apply_dirichlet_bc = (it_nr == 0);
	const FEValuesExtractors::Vector displacement(0);
	const int boundary_id = id_Dirichlet_boundary;

	/*if conditional to check which function should be used for the Dirichlet constraints*/
	if (apply_dirichlet_bc == true)
	{
		VectorTools::interpolate_boundary_values(dof_handler_ref,
												boundary_id,
												ZeroFunction<dim>(dim),
												constraints,
												fe.component_mask(displacement));
	}
	else
	{
		VectorTools::interpolate_boundary_values(dof_handler_ref,
												boundary_id,
												ZeroFunction<dim>(dim),
												constraints,
												fe.component_mask(displacement));	
	}
    constraints.close();
}

template <int dim>
void Solid<dim>::assemble_system()
{
	
	
		std::cout << " Assemble System " << std::flush;

	
	NeoHookeanMaterial<dim> material(this->mu, this->lambda);
	
	//FEValues and FaceValues to compute quantities on quadrature points for our finite
	//element space including mapping from the real cell
	FEValues<dim> fe_values_ref (fe,//The used FiniteElement
								qf_cell,//The quadrature rule for the cell
								update_values| //UpdateFlag for shape function values
								update_gradients| //shape function gradients
								update_JxW_values); //transformed quadrature weights multiplied with Jacobian of transformation 
	FEFaceValues<dim> fe_face_values_ref (fe,
										qf_face, //The quadrature for face quadrature points
										update_values|
										update_normal_vectors| //compute normal vector for face
										update_JxW_values);

	//Quantities to store the local rhs and matrix contribution
	FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
	Vector<double> cell_rhs (dofs_per_cell);
	//Vector with the indicies (global) of the local dofs	
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	//Compute the current, total solution, i.e. starting value of
	//current load step and current solution_delta
	Vector<double> current_solution = get_total_solution(this->solution_delta);
	//Iterators to loop over all active cells
	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active(),
												endc = dof_handler_ref.end();
												
	for(;cell!=endc;++cell)
	{
		//Reset the local rhs and matrix for every cell
		cell_matrix=0.0;
		cell_rhs=0.0;
		//Reinit the FEValues instance for the current cell, i.e.
		//compute the values for the current cell
		fe_values_ref.reinit(cell);
		//Vector to store the gradients of the solution at 
		//n_q_points quadrature points
		std::vector<Tensor<2,dim> > solution_grads_u(n_q_points);
		//Fill the previous vector using get_function_gradients
		fe_values_ref[u_fe].get_function_gradients(current_solution,solution_grads_u);
		//Write the global indicies of the local dofs of the current cell
		cell->get_dof_indices(local_dof_indices);

		//Loop over all quadrature points of the cell
		for(unsigned int k=0; k<n_q_points;++k)
		{
			//BEGIN - INSERT YOUR CODE HERE
			//Compute here the following
			//
			//- deformation gradient using the information in "solution_grads_u[k]" and Physics::Elasticity::StandardTensors<dim>::I
			//- use the deformation gradient to compute
			//- - Kirchhoffstress
			//- - Tangent
			//- Compute the inverse of the Deformation gradient
			Tensor<2,dim> DeformationGradient =  (Tensor<2, dim>(Physics::Elasticity::StandardTensors<dim>::I) + solution_grads_u[k]);
			SymmetricTensor<2,dim> Kirchhoffstress = material.get_KirchhoffStress(DeformationGradient);
			SymmetricTensor<4,dim> Tangent = material.get_Tangent_spt(DeformationGradient);
			Tensor<2,dim> F_inv = invert(DeformationGradient);
			//END - INSERT YOUR CODE HERE
			
			//The quadrature weight for the current quadrature point
			const double JxW = fe_values_ref.JxW(k);
			//Loop over all dof's of the cell
			for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
				//Assemble system_rhs contribution
				//BEGIN - INSERT YOUR CODE HERE
				
				//Compute here the following
				//
				//- the symmetric (SymmetricTensor<2,dim>) gradient with respect to the spatial configuration
				//  for the test function as discussed in the exercise (i) -> use fe_values_ref[u_fe] and F_inv
				//- write into cell_rhs(i)-= the contribution for the current test function at the current quadrature point
				//  !! "-=" due to Newton-Raphson algorithm K\du = -r
				Tensor<2,dim> shape_gradient_wrt_ref_config_i = fe_values_ref[u_fe].gradient(i,k);
				Tensor<2,dim> shape_gradient_wrt_spt_config_i = shape_gradient_wrt_ref_config_i * F_inv;
				SymmetricTensor<2,dim> sym_shape_gradient_wrt_spt_config_i = 
					symmetrize(shape_gradient_wrt_spt_config_i);
				
				cell_rhs(i)-= (sym_shape_gradient_wrt_spt_config_i * Kirchhoffstress) * JxW;
				//END - INSERT YOUR CODE HERE
				
				for(unsigned int j=0; j<dofs_per_cell; ++j)
				{
					//Assemble tangent contribution
					//BEGIN - INSERT YOUR CODE HERE
				
					//Compute here the following
					//
					//- the symmetric (SymmetricTensor<2,dim>) gradient with respect to the spatial configuration
					//  for the ansatz function as discussed in the exercise (j) -> use fe_values_ref[u_fe] and F_inv
					//- write into cell_matrix(i,j) the material and geometrical contributions for the tangent
					Tensor<2,dim> shape_gradient_wrt_ref_config_j = fe_values_ref[u_fe].gradient(j,k);
					Tensor<2,dim> shape_gradient_wrt_spt_config_j =shape_gradient_wrt_ref_config_j * F_inv;
					SymmetricTensor<2,dim> sym_shape_gradient_wrt_spt_config_j = 
						symmetrize(shape_gradient_wrt_spt_config_j);	

					cell_matrix(i,j) += (( symmetrize (transpose(shape_gradient_wrt_spt_config_i) * 
											shape_gradient_wrt_spt_config_j) * Kirchhoffstress ) //geometrical contribution
										+ (sym_shape_gradient_wrt_spt_config_i * Tangent // The material contribution:
											* sym_shape_gradient_wrt_spt_config_j) )
										* JxW;		
					//END - INSERT YOUR CODE HERE					
				}
			}
		}



		//Check for Neumann boundary condition
		for(unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
		{
			if(cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == id_Neumann_boundary )
			{
				fe_face_values_ref.reinit(cell, face);
				
				for(unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
				{
					double step_fraction = double(current_load_step)/double(load_steps);
					double current_load = load_magnitude * step_fraction;
					//Compute the following
					//
					//- The normal vector of the current face at the current quadrature point (fe_face_values_ref)
					//- The normal vector scaled with "current_load"
					const Tensor<1,dim> NormalVector = fe_face_values_ref.normal_vector(f_q_point);
					const Tensor<1,dim> Traction = current_load  * NormalVector;					

					for(unsigned int i = 0; i< dofs_per_cell; ++i)
					{
						//Compute
						//
						//- the test function at the face quadrature point for the i-th test function
						//- write into cell_rhs(i)+= the contribution due to the Neumann boundary condition (-=(-))->+=
						//!! Don't forget the JxW value of that face!!
						const Tensor<1,dim> shape_function = fe_face_values_ref[u_fe].value(i,f_q_point);
						cell_rhs(i)+= (shape_function * Traction) * fe_face_values_ref.JxW(f_q_point);
					}
				}
			}
		}
		//copy local to global
		constraints.distribute_local_to_global(cell_matrix,cell_rhs,
								local_dof_indices,
								tangent_matrix,system_rhs,false);
		

	}
}

template <int dim>
std::pair<unsigned int, double>
Solid<dim>::solve_linear_system(Vector<double> &newton_update)
{

	unsigned int lin_it = 0;
	double lin_res = 0.0;
	std::string solver_type = "CG";
	/*reset the vector newton update*/
	newton_update=0;
	

	std::cout << " SLV " << std::flush;
	if (solver_type == "CG")
	{
		const int solver_its = tangent_matrix.m()
								* multiplier_max_iterations_linear_solver;
		const double tol_sol = 1e-9
								* system_rhs.l2_norm();

		SolverControl solver_control(solver_its, tol_sol);

		GrowingVectorMemory<Vector<double> > GVM;
		SolverCG<Vector<double> > solver_CG(solver_control, GVM);
		PreconditionSSOR<> preconditioner;
		preconditioner.initialize(tangent_matrix, 1.2);
		solver_CG.solve(tangent_matrix,
						newton_update,
						system_rhs,
						preconditioner);
		lin_it = solver_control.last_step();
		lin_res = solver_control.last_value();
	}
	else
	{
		/*throug an error message that the chosen solver type is not implented*/
		Assert (false, ExcMessage("Linear solver type not implemented"));
	}
	/*Write the constraint values into the solution vector (newton-increment) to ensure
	 that these values are used in the sequent*/
	constraints.distribute(newton_update);
	/*Return the number of iterations of the iterative solver and the residual*/
	return std::make_pair(lin_it, lin_res);
}





  template <int dim>
  void Solid<dim>::output_results() const
{
	
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
                                  DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(solution_n,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();
    std::ostringstream filename;
    filename << "solution_loadstep_" << current_load_step << ".vtu";
    std::ofstream output(filename.str().c_str());
    data_out.write_vtu(output);
}


int main ()
{
  using namespace dealii;

	const unsigned int dim=2;

  try
    {
      deallog.depth_console(1);

	  unsigned int loadsteps=10;
	  unsigned int polydegree=1;
	  double load_magnitude=(-7e+3);
	  double mu=70000;
	  double lambda=105000;
      Solid<dim> solid_xd(loadsteps, polydegree, load_magnitude, mu, lambda);
      solid_xd.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}


















