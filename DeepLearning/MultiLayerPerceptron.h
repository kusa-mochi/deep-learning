#pragma once
#include "stdafx.h"
#include "MultiLayerPerceptronCore.h"
#include "Enums.h"

using namespace DeepLearningCore;

namespace DeepLearning
{
	ref class MultiLayerPerceptron
	{
	public:
		MultiLayerPerceptron(
			int numInput,											// 入力の次元数
			cli::array<int>^ numNeuron,								// 各層のニューロンの数
			ActivationFunctionType activationFunctionType,			// 中間層の活性化関数
			OutputActivationFunctionType outputActivationFunctionType	// 出力層の活性化関数
		);
		virtual ~MultiLayerPerceptron();
		cli::array<WEIGHT_TYPE, 2>^ Predict(
			cli::array<WEIGHT_TYPE, 2>^ input,
			cli::array<WEIGHT_TYPE, 2>^ output
		);
	private:
		MultiLayerPerceptronCore* _multiPerceptronCore = NULL;
	};
}
