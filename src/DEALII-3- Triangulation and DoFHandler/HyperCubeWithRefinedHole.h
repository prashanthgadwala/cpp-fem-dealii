#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>


using namespace dealii;

namespace HyperCubeWithRefinedHole
{

//This function refines the mesh around the inner region a number of times 
//according to the given number in the input
template<int dim>
void set_and_execute_refinements(unsigned int number_refinements, Triangulation<dim> &triangulation, types::boundary_id boundary_id_inner_hole)
{
	for(unsigned int i=0; i<number_refinements; i++)
	{
		typename Triangulation<dim>::active_cell_iterator cell= triangulation.begin_active(),
												endc = triangulation.end();
		for(; cell!=endc; ++cell)
		{
			for(unsigned int j=0; j<GeometryInfo<dim>::faces_per_cell; j++)
			{
				/*BEGIN ENTER CODE HERE*/
				if (cell->face(j)->at_boundary() && cell->face(j)->boundary_id() == boundary_id_inner_hole)
				{
					cell->set_refine_flag();
					break;
				}
				/*END ENTER CODE HERE*/
			}
		}
		triangulation.execute_coarsening_and_refinement();		
	}
}



template<int dim>
void set_boundary_ids(Triangulation<dim> &triangulation, unsigned int id_Dirichlet, unsigned int id_Neumann, double outer_radius)
{
	typename Triangulation<dim>::active_cell_iterator cell= triangulation.begin_active(),
											endc = triangulation.end();

	for(; cell!=endc; ++cell)
	{
		for(unsigned int j=0; j<GeometryInfo<dim>::faces_per_cell; j++)
		{
			/*BEGIN ENTER CODE HERE*/
			if (cell->face(j)->at_boundary() && (std::abs((cell->face(j)->center()[0] - outer_radius)) < 1e-10) )
			{
				cell->face(j)->set_boundary_id(id_Neumann);
				std::cout<<"Face at Neumann boundary at:"<<cell->face(j)->center()<<std::endl;
			}
			else if (cell->face(j)->at_boundary() && std::abs((cell->face(j)->center()[0] + outer_radius)) < 1e-10)
			{
				cell->face(j)->set_boundary_id(id_Dirichlet);
				std::cout<<"Face at Dirichlet boundary at:"<<cell->face(j)->center()<<std::endl;
			}
			/*END ENTER CODE HERE*/
		}
		
	}	
}
//This function uses the GridGenerator to generate a
//hypercupe with zylindrical hole
template<int dim>
void generate_grid(Triangulation<dim> &triangulation,
							unsigned int int_nbr_refinements,
							unsigned int id_dirichelt_boundary,
						  unsigned int id_neumann_boundary )
{
	const double outer_radius = 1.0;
	const double inner_radius = 0.5;
	const Point<dim> center;

	GridGenerator::hyper_cube_with_cylindrical_hole(triangulation,
													inner_radius,
												 outer_radius,
												 0.5,
												 1,
												 false /*boundary_id_inner_hole is set to 1*/);
	
	std::unique_ptr<Manifold<dim>> ptr_manifold=nullptr;
	
    if(dim==2)
    {
        ptr_manifold = std::make_unique<SphericalManifold<dim>>(center);
    }
    else if(dim==3)
    {
        ptr_manifold = std::make_unique<CylindricalManifold<dim>>(dim-1);
    }
	else
	{
		throw std::runtime_error("only allowed for dim == 2 or dim == 3");
	}
	types::boundary_id boundary_id_inner_hole=1;
	types::manifold_id manifold_id_inner_hole=1;
	//Set the manifold id of all boundary faces and edges with given boundary id 
	triangulation.set_all_manifold_ids_on_boundary(boundary_id_inner_hole,manifold_id_inner_hole);
	/*If this is not done and the manifold_id equals number::invalid_manifold_id (which is default)
	*the triangulation object queries the boundary_id if the face is at the boundary or the material_id
	*/
    triangulation.set_manifold (manifold_id_inner_hole, *ptr_manifold);
	
	triangulation.refine_global(1);
	set_and_execute_refinements(int_nbr_refinements, triangulation, boundary_id_inner_hole);
	triangulation.reset_manifold(manifold_id_inner_hole);
		
	set_boundary_ids(triangulation,id_dirichelt_boundary, id_neumann_boundary, outer_radius);
}




}//end of namespace
