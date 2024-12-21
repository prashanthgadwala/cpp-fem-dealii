
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

/*! \brief Coding Assignment 3
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
	unsigned int nbr_adaptive_refinements = 1;
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
	//-----------more to come
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

	/*BEGIN ENTER CODE HERE*/
 	DoFRenumbering::Cuthill_McKee(dof_handler_ref);
// 	DoFRenumbering::random(dof_handler_ref);
	/*END ENTER CODE HERE*/
	
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
	
	/*BEGIN ENTER CODE HERE*/
	tangent_matrix.reinit (sparsity_pattern);
	system_rhs.reinit(n_dofs_u);
	solution_delta.reinit(n_dofs_u);
	solution_n.reinit(n_dofs_u);
	/*END ENTER CODE HERE*/
}


template <int dim>
void Solid<dim>::assemble_system()
{
}



template <int dim>
void Solid<dim>::solve_load_step_NR(Vector<double> &solution_delta)
{
}



template <int dim>
void Solid<dim>::make_constraints(const int &it_nr)
{


}







  template <int dim>
  void Solid<dim>::output_results() const
{
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


















