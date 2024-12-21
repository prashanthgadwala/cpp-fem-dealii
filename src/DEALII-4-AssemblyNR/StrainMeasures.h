#ifndef STRAINMEASURES_H
#define STRAINMEASURES_H

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <iostream>

using namespace dealii;

/*! \brief Functions to compute several strain measures grouped in a namespace
 *  *  \author Dominic Soldner
 * 	\version 3.0
 * \date Jan 2021
 * 
 * A namespace that groups several functions to compute different stress measures, 
 * using a 2nd order tensor, i.e. the deformation gradient,
 * as input. Standard unit tensors are used from the header file
 * deal.II/physics/elasticity/standard_tensors.h.
 *
*/

/* A namespace that groups functions to compute several strain measures, 
 * using a 2nd order tensor, i.e. the deformation gradient,
 * as input
 */
namespace StrainMeasures
{
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------

	/*!Compute the right CauchyGreen strain tensor as
	 * \f$ \mathbf{C} =  \mathbf{F}^T \cdot \mathbf{F} \f$
	 * @param F Deformation gradient
	 */
	template <int dim>
	SymmetricTensor<2, dim> get_RightCauchyGreenTensor(const Tensor<2, dim> &F)
	{
		/* Compute the right CauchyGreen tensor
		* using the member variable "F" and the deal.II function
		* "transpose(). Hint: in order to get a symmetric tensor 
		* use the function "symmetrize()"!
		*/
		SymmetricTensor<2,dim> RightCauchyGreenTensor;
		
		//BEGIN - INSERT YOUR CODE HERE
		RightCauchyGreenTensor = symmetrize( transpose(F) * F);
		
		
		//END - INSERT YOUR CODE HERE	
		
		return RightCauchyGreenTensor;
	}
	//------------------------------------------
	/*!Compute the left CauchyGreen strain tensor as
	 * \f$ \mathbf{b} =  \mathbf{F} \cdot \mathbf{F}^T \f$
	 * @param F Deformation gradient
	 */
	template <int dim>
	SymmetricTensor<2, dim> get_LeftCauchyGreenTensor(const Tensor<2, dim> &F) 
	{
		/* Compute the left CauchyGreen tensor
		* using the member variable "F" and the deal.II function
		* "transpose(). Hint: in order to get a symmetric tensor 
		* use the function "symmetrize()"!
		*/
		SymmetricTensor<2,dim> LeftCauchyGreenTensor;
		
		//BEGIN - INSERT YOUR CODE HERE
		LeftCauchyGreenTensor = symmetrize(F * transpose(F));
		
		
		
		//END - INSERT YOUR CODE HERE	
		
		return LeftCauchyGreenTensor;
	}
	//------------------------------------------
	/*!Compute the Green Lagrange strain tensor as
	 * \f$ \mathbf{E} =  \frac{1}{2} \left[ \mathbf{C} - \mathbf{I} \right] \f$
	 * @param F Deformation gradient
	 */
	template <int dim>
	SymmetricTensor<2, dim> get_GreenLagrangeTensor(const Tensor<2, dim> &F) 
	{
		/* Compute the Green tensor
		* using the previously definted function 
		* "get_RightCauchyGreenTensor" and "Physics::Elasticity::StandardTensors<dim>::I"
		*/
		SymmetricTensor<2,dim> GreenLagrangeTensor;
		
		//BEGIN - INSERT YOUR CODE HERE
		GreenLagrangeTensor = (0.5 * (get_RightCauchyGreenTensor(F) - Physics::Elasticity::StandardTensors<dim>::I ) );
		
		
		//END - INSERT YOUR CODE HERE	
		
		return GreenLagrangeTensor;
	}
	//------------------------------------------
	//------------------------------------------
	/*!Compute the Almansi strain tensor as
	 * \f$ \mathbf{e} =  \frac{1}{2} \left[ \mathbf{I} - \mathbf{b}^{-1} \right] \f$
	 * @param F Deformation gradient
	 */
	template <int dim>
	SymmetricTensor<2, dim> get_AlmansiTensor(const Tensor<2, dim> &F) 
	{
		/* Compute the Almansi strain tensor
		* using the previously definted function 
		* "get_LeftCauchyGreenTensor", "Physics::Elasticity::StandardTensors<dim>::I" and the 
		* deal.II function "invert()" for the tensor LCG
		*/
		SymmetricTensor<2,dim> AlmansiTensor;
		//BEGIN - INSERT YOUR CODE HERE
		AlmansiTensor = (0.5 * (Physics::Elasticity::StandardTensors<dim>::I - invert(get_LeftCauchyGreenTensor(F) ) ));
		
		
		//END - INSERT YOUR CODE HERE	
		
		return AlmansiTensor;
	}
	//------------------------------------------

    //------------------------------------------
    /*! A function to compute and return the determinant
    * of the deformation gradient \f$ J =
    * \text{det}\left( \mathbf{F} \right) \f$
    * @return \f$ J = \text{det}\left( \mathbf{F} \right) \f$ 
    */	
    template <int dim>
    double get_DeterminantDefoGrad(const Tensor<2, dim> &F)
    {
        double det_F = determinant(F);
        if(det_F <= 0)
        {
            std::cout<<"F:\n"<<F<<std::endl;
            throw std::runtime_error("det_F !> 0");
        }
        return det_F;
    }
}


#endif