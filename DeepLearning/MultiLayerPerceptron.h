#pragma once
#include "stdafx.h"
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
			ActivationFunctionType activationFunctionType,			// 中間層の活性化関数
			OutputActivationFunctionType outputActivationFunctionType	// 出力層の活性化関数
		);
		virtual ~MultiLayerPerceptron();
		void SetWeights(cli::array<cli::array<WEIGHT_TYPE, 2>^>^ weights);
		void SetBias(cli::array<cli::array<WEIGHT_TYPE>^>^ bias);
		cli::array<WEIGHT_TYPE, 2>^ Predict(cli::array<WEIGHT_TYPE, 2>^ input);
	private:
		int _numInput = 0;
		cli::array<int>^ _numNeuron = nullptr;
		MultiLayerPerceptronCore * _multiLayerPerceptronCore = NULL;
		void ManagedArray2NativeArray(cli::array<WEIGHT_TYPE, 2>^ input, WEIGHT_TYPE*** output);
		cli::array<WEIGHT_TYPE, 2>^ NativeArray2ManagedArray(WEIGHT_TYPE** input, int rows, int cols);
	};
}
