#pragma once

#ifdef EXPORTING_
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC __declspec(dllimport)
#endif

typedef double WEIGHT_TYPE;
#define ARGUMENT_EXCEPTION 1
#define ARGUMENT_NULL_EXCEPTION 2
#define INVALID_OPERATION_EXCEPTION 3
#define FUNCTION_NONE 0
#define FUNCTION_SIGMOID 1
#define FUNCTION_RELU 2
#define FUNCTION_SOFTMAX 3

#ifdef EXPORTING_
#define EIGEN_NO_DEBUG		// コード内のassertを無効化．
#define EIGEN_MPL2_ONLY		// LGPLライセンスのコードを使わない．
#include <iostream>
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;
typedef Matrix<WEIGHT_TYPE, -1, -1> MatrixXX;
typedef Matrix<WEIGHT_TYPE, 1, -1> VectorXX;
typedef struct ST_LayerBackwardOutput
{
	MatrixXX x;
	MatrixXX y;
} LayerBackwardOutput;
#endif
