
/*
 *
 * Author: Dominic Soldner, 2021
 * 
 * 
 * In this Coding Assignment the previously covered tensor class
 * templates are used to compute different strain and stress measures 
 * for Neo-Hookean material (hyperelasticity).
 * For now a deformation gradient F is given and passed to an instance
 * of the class "NeoHookeanMaterial".
 * Author: Dominic Soldner, FAU Erlangen-Nuremberg, 2021
 
 */


#include "StrainMeasures.h"
#include "NeoHookeanMaterial.h"

#include <iostream>
#include <deal.II/base/tensor.h>



//----------------------------------------------------

int main ()
{
   
    
    const int dim=3;
   
	//Prescribed material parameters
	double mu=1;
	double lambda=1;
	//Create an instance of the class template "NeoHookeanMaterial"
	NeoHookeanMaterial<dim> testmaterial(mu,lambda);
	
	/*Change the values of the deformation gradient in order
	 * to compute the various strain and stress measures
	 * used during the excercise
	 */
	Tensor<2,dim> DefoGrad;
	DefoGrad[0][0] = 0; 	DefoGrad[0][1] = -2; 	DefoGrad[0][2] = 0;
	DefoGrad[1][0] = 0.5; 	DefoGrad[1][1] = 0; 	DefoGrad[0][2] = 0;
	DefoGrad[2][0] = 0; 	DefoGrad[2][1] = 0; 	DefoGrad[2][2] = 1;
	
	std::cout<<"\n\n\t\tCodingAssignment 2 \n\n"<<std::endl;
	
	std::cout<<"DefoGrad: \t"<<DefoGrad<<std::endl;
	/* Output the strain and stress measusres of interest
	 * as shown in the last coding assignment and check
	 * the values.
	 * A few examples are given in the following
	 */
	std::cout<<"LeftCauchyGreenTensor: \t"<<StrainMeasures::get_LeftCauchyGreenTensor(DefoGrad)<<std::endl;
	std::cout<<"RightCauchyGreenTensor: \t"<<StrainMeasures::get_RightCauchyGreenTensor(DefoGrad)<<std::endl;
	std::cout<<"GreenTensor: \t"<<StrainMeasures::get_GreenLagrangeTensor(DefoGrad)<<std::endl;
	std::cout<<"AlmansiTensor: \t"<<StrainMeasures::get_AlmansiTensor(DefoGrad)<<std::endl;

	std::cout<<"CauchyStressTensor: \t"<<testmaterial.get_CauchyStress(DefoGrad)<<std::endl;
	std::cout<<"KirchhoffStress: \t"<<testmaterial.get_KirchhoffStress(DefoGrad)<<std::endl;
	std::cout<<"PiolaStress: \t"<<testmaterial.get_PiolaStress(DefoGrad)<<std::endl;
	std::cout<<"2ndPiolaKirchhoffStress: \t"<<testmaterial.get_2ndPiolaKirchhoffStress(DefoGrad)<<std::endl;

	std::cout<<"\n\n"<<std::endl;

    //------------------------------------------------
    
}

//----------------------------------------------------
