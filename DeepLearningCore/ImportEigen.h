#pragma once

// Eigenライブラリの読込
#define EIGEN_NO_DEBUG		// コード内のassertを無効化．
#define EIGEN_MPL2_ONLY		// LGPLライセンスのコードを使わない．
#include "Eigen/Dense"
#include "CommonTypes.h"
using namespace Eigen;
typedef Matrix<WEIGHT_TYPE, -1, -1> MatrixXX;
typedef Matrix<WEIGHT_TYPE, 1, -1> VectorXX;
//typedef Matrix<WEIGHT_TYPE, -1, 1> VerticalVectorXX;

#ifdef _DEBUG
#define PRINT_MATRIX(m,name) (std::cout << name << std::endl << m << std::endl)
#else
#define PRINT_MATRIX(m,name)
#endif
