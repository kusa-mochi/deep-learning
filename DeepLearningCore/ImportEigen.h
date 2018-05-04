#pragma once

// Eigen���C�u�����̓Ǎ�
#define EIGEN_NO_DEBUG		// �R�[�h����assert�𖳌����D
#define EIGEN_MPL2_ONLY		// LGPL���C�Z���X�̃R�[�h���g��Ȃ��D
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
