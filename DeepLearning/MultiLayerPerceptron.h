#pragma once
#include "stdafx.h"
#include "CommonTypes.h"
#include "MultiLayerPerceptronCore.h"
#include "Enums.h"

using namespace DeepLearningCore;

namespace DeepLearning
{
	public ref class MultiLayerPerceptron
	{
	public:
		MultiLayerPerceptron(
			int numInput,											// 入力の次元数
			cli::array<int>^ numNeuron,								// 各層のニューロンの数
			cli::array<LayerType>^ layerType						// 各層の計算方法
		);
		virtual ~MultiLayerPerceptron();
		void SetWeights(cli::array<cli::array<WEIGHT_TYPE, 2>^>^ weights);
		void SetBias(cli::array<cli::array<WEIGHT_TYPE>^>^ bias);
		cli::array<WEIGHT_TYPE, 2>^ Predict(cli::array<WEIGHT_TYPE, 2>^ input);
	private:
		int _numInput = 0;
		MultiLayerPerceptronCore * _multiLayerPerceptronCore = NULL;
		void ManagedArray2NativeArray(cli::array<WEIGHT_TYPE, 2>^ input, WEIGHT_TYPE*** output);
		cli::array<WEIGHT_TYPE, 2>^ NativeArray2ManagedArray(WEIGHT_TYPE** input, int rows, int cols);
	};
}
