#pragma once

#define _EXPORTING

#ifdef _EXPORTING  
#define CLASS_DECLSPEC    __declspec(dllexport)  
#else  
#define CLASS_DECLSPEC    __declspec(dllimport)  
#endif  

#define EIGEN_NO_DEBUG		// コード内のassertを無効化．
#define EIGEN_MPL2_ONLY		// LGPLライセンスのコードを使わない．
#include <vector>
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

typedef double WEIGHT_TYPE;
typedef Matrix<WEIGHT_TYPE, -1, -1> MatrixXX;

#define ARGUMENT_EXCEPTION 1
#define ARGUMENT_NULL_EXCEPTION 2
//#define SIGMOID(x) (1.0/(1.0+(-x).array().exp()))	// x:MatrixXX
//#define RELU(x) ((x.array() > 0.0).all() ? x : MatrixXX::Zero(x.size()))	// x:MatrixXX
//#define ACTIVATION_FUNCTION(x) SIGMOID(x)			// x:MatrixXX
