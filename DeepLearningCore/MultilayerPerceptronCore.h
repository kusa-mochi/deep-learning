#pragma once

#include "DeepLearningCore.h"

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
		void SetWeights(WEIGHT_TYPE*** weights);
		void SetBias(WEIGHT_TYPE** bias);
		void Predict(WEIGHT_TYPE** input, int numData, WEIGHT_TYPE*** output);
	private:
		int _numLayer = 0;
		int _numInput = 0;
		int* _numNeuron = NULL;
	private:
		WEIGHT_TYPE(*_ActivationFunction)(WEIGHT_TYPE) = NULL;
		WEIGHT_TYPE(*_OutputActivationFunction)(WEIGHT_TYPE) = NULL;
#ifdef EXPORTING_
		MatrixXX* _weight = NULL;
		VectorXX* _bias = NULL;
		void InitializeWeights();
		MatrixXX Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols);
		void Matrix2Pointer(MatrixXX m, WEIGHT_TYPE*** p);
		MatrixXX MatrixPlusVector(MatrixXX m, VectorXX v);
#else
		// 以下は，このクラスのインスタンスをdeleteする際に，
		// 解放対象とするメモリ領域をヒープ領域のサイズに一致させるための措置。
		// インポートする側のアプリでは特に気にする必要はない。
		int* _weight = NULL;
		int* _bias = NULL;
		void InitializeWeights();
		void Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols);
		void Matrix2Pointer(int m, WEIGHT_TYPE*** p);
		void MatrixPlusVector(int m, int v);
#endif
	};
}
