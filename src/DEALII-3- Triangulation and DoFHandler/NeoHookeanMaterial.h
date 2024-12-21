#ifndef NEOHOOKEANMATERIAL_H
#define NEOHOOKEANMATERIAL_H

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "StrainMeasures.h"

#include <iostream>

using namespace dealii;

//Detailed Description of the Class NeoHookeanMaterial in documentation
//-----------------------------------------------------------
//-----------------------------------------------------------
/*! \brief Hyperelastic Material of Neo-Hookean type
 *  *  \author Dominic Soldner
 * 	\version 3.0
 * \date Jan 2021
 * 
* A class to compute several stress measures, 
 * using a 2nd order tensor, i.e. the deformation gradient,
 * as input
 *
* 
* Once an instance of this object is created, i.e.
* ~~~~~~~~~~~~~~~~~~~~~~{.c}
* const int dim=3;
* double lambda = 1;
* double mu = 1;
* Tensor<2,dim> DefoGrad;
* DefoGrad[0][0] = 0; 	DefoGrad[0][1] = -2; 	DefoGrad[0][2] = 0;
* DefoGrad[1][0] = 0.5; 	DefoGrad[1][1] = 0; 	DefoGrad[0][2] = 0;
* DefoGrad[2][0] = 0; 	DefoGrad[2][1] = 0; 	DefoGrad[2][2] = 1;
* 
* NeoHookeanMaterial<dim> Material(mu, lambda);
* 
* ~~~~~~~~~~~~~~~~~~~~~~
* The quantities can be called via
* ~~~~~~~~~~~~~~~~~~~~~~{.c}
* Tensor<2,dim> Sigma;
* Sigma = Material.get_CauchyStress(DefoGrad) ;
* ~~~~~~~~~~~~~~~~~~~~~~ 
*/

//Declaration of the class template
//-----------------------------------------------------------
//-----------------------------------------------------------
/* A class to compute several stress measures, 
 * using a 2nd order tensor, i.e. the deformation gradient,
 * as input
 */
template<int dim>
class NeoHookeanMaterial 
{
    public:
		/*! Hyperelastic material of Neo-Hookean type.
		 * The strain energy density is given as
		 * \f$ \Psi^{NH}\left( \mathbf{C} \right) 
		 * = \frac{\mu}{2} \left[ I_C - 3 \right] - 
		 * \mu {ln}\left( J\right) + \frac{\lambda}{2}
		 * {ln}^2 \left( J \right) \f$
		 * @param mu The Lame parameter \f$ \mu \f$
		 * @param lambda The Lame parameter \f$ \lambda \f$
		 */		
        NeoHookeanMaterial(double mu, double lambda);
        ~NeoHookeanMaterial(){}
         /*! A function to compute and return the Kirchhoff stress \f$ \boldsymbol{\tau} 
		 * = \text{det} \left( \mathbf{F} \right) \boldsymbol{\sigma} 
		 * = J \boldsymbol{\sigma} \f$
		 * @return The Kirchhoff stress \f$ \boldsymbol{\tau}\f$
		 */				
        SymmetricTensor<2, dim> get_KirchhoffStress(const Tensor<2, dim> &F) ;
		 /*! A function to compute and return the \f$ 2^{\text{nd}}\f$ Piola-Kirchhoff stress
		 * \f$ \mathbf{S} =  J \mathbf{F}^{-1} \cdot \boldsymbol{\sigma} 
		 * \cdot \mathbf{F}^{-t} \f$
		 * @return \f$ 2^{\text{nd}}\f$ Piola-Kirchhoff stress \f$ \mathbf{S} \f$
		 */	
		SymmetricTensor<2, dim> get_2ndPiolaKirchhoffStress(const Tensor<2, dim> &F);
				
        /*! A function to compute and return the Cauchy stress
		 * \f$ \boldsymbol{\sigma} =  \frac{\mu}{J} \left[ \mathbf{b} - \mathbf{I} \right]
		 * + \frac{\lambda \text{ln}\left( J \right)}{J} \mathbf{I} \f$
		 * , with \f$ \mathbf{b} = \mathbf{F} \cdot \mathbf{F}^t\f$
		 * @return Cauchy stress \f$ \boldsymbol{\sigma} \f$
		 */		
        SymmetricTensor<2, dim> get_CauchyStress(const Tensor<2, dim> &F) ;
		/*! A function to compute and return the Piola stress
		 * \f$ \mathbf{P} =  J \cdot \boldsymbol{\sigma} \cdot \mathbf{F}^{-t}\f$
		 * @return Piola stress \f$ \mathbf{P} \f$
		 */			
        Tensor<2, dim> get_PiolaStress(const Tensor<2, dim> &F) ;
		
		SymmetricTensor<4, dim> get_Tangent_spt(const Tensor<2, dim> &F) ;
    protected:

    private:
		
		// Lame and material parameters
		/*! The Lame parameter \f$ \mu \f$
		 */
        const double mu;
		/*! The Lame parameter \f$ \lambda \f$
		 */		
		const double lambda;     
};




//Definition of the class template
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
template <int dim> //Constructor
NeoHookeanMaterial<dim>::NeoHookeanMaterial(double mu, double lambda)
:
mu(mu),
lambda(lambda)
{	
}
//------------------------------------------






//------------------------------------------
template <int dim>
SymmetricTensor<2, dim> NeoHookeanMaterial<dim>::get_CauchyStress(const Tensor<2, dim> &F)
{
	/* Implement a routine to compute the Cauchy stress using 
	 * the function "get_LeftCauchyGreenTensor" and the unit
	 * tensor "Physics::Elasticity::StandardTensors<dim>::I". Be aware that the 
	 * the material parameters are member variables of this 
	 * class, i.e. you can simple write for instance
	 * mu * 2 to multiply the variable mu with 2. As always
	 * there is no harm in writing this->mu to clarify that you 
	 * intend to use the member variable of this class.
	 * Furthermore, use the function "get_DeterminantDefoGrad" of the namespace StrainMeasures
	 * in order to compute the determinant of the deformation gradient
	 * since it contains a check to see if its value is greater zero.
	 */ 
	SymmetricTensor<2,dim> CauchyStress;
	
	//BEGIN - INSERT YOUR CODE HERE
	double det_F = StrainMeasures::get_DeterminantDefoGrad(F);
	CauchyStress = (   (mu/det_F)*(StrainMeasures::get_LeftCauchyGreenTensor(F)
		- Physics::Elasticity::StandardTensors<dim>::I)
		+ ( ((lambda * std::log(det_F))/det_F ) * Physics::Elasticity::StandardTensors<dim>::I)  );
	
	
    //END - INSERT YOUR CODE HERE
	
    return CauchyStress;
}

//------------------------------------------

template <int dim>
Tensor<2, dim> NeoHookeanMaterial<dim>::get_PiolaStress(const Tensor<2, dim> &F)
{
	/*
	 * Compute the Piola stress based on the cauchy stress.
	 * The function "get_CauchyStress" 
	 * returns a SymmetricTensor of 2nd order. Due to the 
	 * way this class is defined in deal.II, direcly using
	 * this tensor to compute the Piola stress tensor would
	 * lead to a symmetric tensor which it should not.
	 * Therefore the "static_cast" operator is used to transform
	 * the SymmetricTensor to a regular 2nd order tensor.
	 */
	Tensor<2,dim> PiolaStress;
    Tensor<2,dim> CauchyStress = static_cast<Tensor<2,dim> > ( get_CauchyStress(F) );

	//BEGIN - INSERT YOUR CODE HERE
	double det_F = StrainMeasures::get_DeterminantDefoGrad(F);
	Tensor<2,dim> F_inv = invert(F);
	PiolaStress = det_F * CauchyStress * (transpose(F_inv));
	
	
    //END - INSERT YOUR CODE HERE	

    return PiolaStress;
	
}

template <int dim>
SymmetricTensor<2, dim> NeoHookeanMaterial<dim>::get_2ndPiolaKirchhoffStress(const Tensor<2, dim> &F)
{
	/* Implement a routine to compute the 2nd PiolaKirchhoff stress using 
	 * the function "get_CauchyStress" and the "static_cast<Tensor<2,dim>> 
	 * similar to the previous example.
	 */ 
    SymmetricTensor<2,dim> SecPiolaKirchhoffStress;

	//BEGIN - INSERT YOUR CODE HERE
	double det_F = StrainMeasures::get_DeterminantDefoGrad(F);
	Tensor<2,dim> F_inv = invert(F);
	SecPiolaKirchhoffStress =  symmetrize(  det_F * F_inv * static_cast<Tensor<2,dim> >( get_CauchyStress(F)) * transpose (F_inv) );
	
	
    //END - INSERT YOUR CODE HERE	

    return SecPiolaKirchhoffStress;
	
}

//------------------------------------------
template <int dim>
SymmetricTensor<2, dim> NeoHookeanMaterial<dim>::get_KirchhoffStress(const Tensor<2, dim> &F)
{
	/* This function returns the Kirchhoff stress "tau".
	* Use the already available function "get_CauchyStress"
	* in order to compute this quantity.
	*/
	
	SymmetricTensor<2,dim> KirchhoffStress;
	
	//BEGIN - INSERT YOUR CODE HERE
	KirchhoffStress = get_CauchyStress(F) * StrainMeasures::get_DeterminantDefoGrad(F);
	
	
    //END - INSERT YOUR CODE HERE	
	
    return KirchhoffStress;
}

//------------------------------------------

template <int dim>
SymmetricTensor<4, dim> NeoHookeanMaterial<dim>::get_Tangent_spt(const Tensor<2, dim> &F)
{
	double det_F = StrainMeasures::get_DeterminantDefoGrad(F);
    return ( ( (lambda/det_F)*Physics::Elasticity::StandardTensors<dim>::IxI
	 + 2*( (mu-(lambda*std::log(det_F)))/ det_F  )*Physics::Elasticity::StandardTensors<dim>::S   )
		* det_F);
}
//END PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------

#endif