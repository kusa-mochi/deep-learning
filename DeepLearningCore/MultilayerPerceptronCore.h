#pragma once

#ifdef EXPORTING_
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC __declspec(dllimport)
#endif

#include "CommonTypes.h"

#include <iomanip>
using namespace std;

#ifdef EXPORTING_
#include "Layer.h"
#include "IdentityLayerCore.h"
#include "AddLayerCore.h"
#include "MulLayerCore.h"
#include "AffineLayerCore.h"
#include "SigmoidLayerCore.h"
#include "ReLULayerCore.h"
#include "SoftmaxWithLoss.h"
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
		void Learn(WEIGHT_TYPE** input, WEIGHT_TYPE** teachData, int numData, double learningRate = 0.3);
#ifdef _DEBUG
		void DebugGradient(WEIGHT_TYPE** x, WEIGHT_TYPE** t, int numData, WEIGHT_TYPE**** outputWeights, WEIGHT_TYPE**** outputBias);
		void DebugNumericGradient(WEIGHT_TYPE** x, WEIGHT_TYPE** t, int numData, WEIGHT_TYPE**** outputWeights, WEIGHT_TYPE**** outputBias);
#endif
	private:
		int _numLayer = 0;
		int _numInput = 0;
		double _learningRate = 0.3;
	private:
#ifdef EXPORTING_
		MatrixXX* _weight = NULL;
		VectorXX* _bias = NULL;
		LayerInfo* _layerInfo = NULL;
		Layer* _layer = NULL;
		Layer* _layerEnd = NULL;
		Layer* _lastLayer = NULL;
		void InitializeWeights();
		void InitializeLayers();
		void InitializeLastLayer();
		MatrixXX Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols);
		void Matrix2Pointer(MatrixXX m, WEIGHT_TYPE*** p);
		MatrixXX MatrixPlusVector(MatrixXX m, VectorXX v);
		MatrixXX PredictCore(MatrixXX input);
		void LearnCore(MatrixXX input, MatrixXX teach, double learningRate);
		MatrixXX ApplyLastLayer(MatrixXX m, MatrixXX t);
		MatrixXX Loss(MatrixXX x, MatrixXX t);
		WeightsAndBias Gradient(MatrixXX x, MatrixXX t);
		WeightsAndBias NumericGradient(MatrixXX x, MatrixXX t);
#else
		// 以下は，このクラスのインスタンスをdeleteする際に，
		// 解放対象とするメモリ領域をヒープ領域のサイズに一致させるための措置。
		// インポートする側のアプリでは特に気にする必要はない。
		int* _weight = NULL;
		int* _bias = NULL;
		LayerInfo* _layerInfo = NULL;
		int* _layer = NULL;
		int* _layerEnd = NULL;
		int* _lastLayer = NULL;
		void InitializeWeights();
		void InitializeLayers();
		void InitializeLastLayer();
		int Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols);
		void Matrix2Pointer(int m, WEIGHT_TYPE*** p);
		int MatrixPlusVector(int m, int v);
		int PredictCore(int input);
		void LearnCore(int input, int teach);
		int ApplyLastLayer(int m, int t);
		int Loss(int m, int t);
		int Gradient(int m, int t);
		int NumericGradient(int m, int t);
#endif
	};
}
