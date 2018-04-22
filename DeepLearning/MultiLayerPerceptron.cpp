#include "stdafx.h"
#include "MultiLayerPerceptron.h"


namespace DeepLearning
{
	WEIGHT_TYPE Sigmoid(WEIGHT_TYPE x)
	{
		return (1.0 / (1.0 + std::exp(x)));
	}

	MultiLayerPerceptron::MultiLayerPerceptron(
		int numInput,											// 入力の次元数
		cli::array<int>^ numNeuron,								// 各層のニューロンの数
		ActivationFunctionType activationFunctionType,			// 中間層の活性化関数
		OutputActivationFunctionType outputActivationFunctionType		// 出力層の活性化関数
	)
	{
		int* numNeuronPointer = new int[numNeuron->Length];
		for (int i = 0; i < numNeuron->Length; i++) numNeuronPointer[i] = numNeuron[i];

		_multiPerceptronCore = new MultiLayerPerceptronCore(
			numInput,
			numNeuron->Length,
			numNeuronPointer,
			FUNCTION_SIGMOID,	// TODO: コンストラクタの入力により値を決定する。
			FUNCTION_NONE		// TODO: コンストラクタの入力により値を決定する。
		);
	}


	MultiLayerPerceptron::~MultiLayerPerceptron()
	{
	}

	cli::array<WEIGHT_TYPE, 2>^ MultiLayerPerceptron::Predict(cli::array<WEIGHT_TYPE, 2>^ input, cli::array<WEIGHT_TYPE, 2>^ output)
	{
		return nullptr;	// TODO
	}
}
