#pragma once

#ifdef EXPORTING_
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC __declspec(dllimport)
#endif

#include <iostream>
using namespace std;

#include "CommonTypes.h"

#ifdef EXPORTING_
#include "Layer.h"
#include "IdentityLayerCore.h"
#include "AddLayerCore.h"
#include "MulLayerCore.h"
#include "AffineLayerCore.h"
#include "SigmoidLayerCore.h"
#include "ReLULayerCore.h"
#endif

namespace DeepLearningCore
{
	class DECLSPEC MultiLayerPerceptronCore
	{
	public:
		MultiLayerPerceptronCore(
			int numInput,					// 入力の次元数
			int numLayer,					// 層数
			LayerInfo* layerInfo			// 各層のニューロン数，計算方法などの情報
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
			return _layerInfo[_numLayer - 1].NumNeuron;
		}
		void SetWeights(WEIGHT_TYPE*** weights);
		void SetBias(WEIGHT_TYPE** bias);
		void Predict(WEIGHT_TYPE** input, int numData, WEIGHT_TYPE*** output);
	private:
		int _numLayer = 0;
		int _numInput = 0;
	private:
#ifdef EXPORTING_
		MatrixXX* _weight = NULL;
		VectorXX* _bias = NULL;
		LayerInfo* _layerInfo = NULL;
		Layer* _layer = NULL;
		void InitializeWeights();
		void InitializeLayers();
		MatrixXX Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols);
		void Matrix2Pointer(MatrixXX m, WEIGHT_TYPE*** p);
		MatrixXX MatrixPlusVector(MatrixXX m, VectorXX v);
		
#else
		// 以下は，このクラスのインスタンスをdeleteする際に，
		// 解放対象とするメモリ領域をヒープ領域のサイズに一致させるための措置。
		// インポートする側のアプリでは特に気にする必要はない。
		int* _weight = NULL;
		int* _bias = NULL;
		LayerInfo* _layerInfo = NULL;
		int* _layer = NULL;
		void InitializeWeights();
		void Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols);
		void Matrix2Pointer(int m, WEIGHT_TYPE*** p);
		void MatrixPlusVector(int m, int v);
#endif
	};
}
