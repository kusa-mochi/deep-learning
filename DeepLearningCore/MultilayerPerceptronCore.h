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
#endif

namespace DeepLearningCore
{
	class DECLSPEC MultiLayerPerceptronCore
	{
	public:
		MultiLayerPerceptronCore(
			int numInput,					// 入力の次元数
			int numLayer,					// 層数
			int* numNeuron,					// 各層のニューロンの数
			int activationFunctionType,		// 中間層の活性化関数の種類
			int outputActivationFunctionType	// 出力層の活性化関数
		);
		virtual ~MultiLayerPerceptronCore();
		int GetNumLayer()
		{
			return _numLayer;
		}
		int GetNumInput()
		{
			return _numInput;
		}
		int GetNumOutput()
		{
			return _numNeuron[_numLayer - 1];
		}
		void Predict(WEIGHT_TYPE** input, int numData, WEIGHT_TYPE** output);
	private:
		int _numLayer = 0;
		int _numInput = 0;
		int* _numNeuron;
#ifdef EXPORTING_
	private:
		WEIGHT_TYPE(*_ActivationFunction)(WEIGHT_TYPE) = NULL;
		WEIGHT_TYPE(*_OutputActivationFunction)(WEIGHT_TYPE) = NULL;
		MatrixXX* _weight;
		void InitializeWeights();
		MatrixXX Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols);
		WEIGHT_TYPE** Matrix2Pointer(MatrixXX m);
#endif
	};
}
