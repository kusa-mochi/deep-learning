#include "stdafx.h"
#include "MultiLayerPerceptron.h"


namespace DeepLearning
{
	WEIGHT_TYPE Sigmoid(WEIGHT_TYPE x)
	{
		return (1.0 / (1.0 + std::exp(x)));
	}

	MultiLayerPerceptron::MultiLayerPerceptron(
		int numInput,											// “ü—Í‚ÌŸŒ³”
		cli::array<int>^ numNeuron,								// Še‘w‚Ìƒjƒ…[ƒƒ“‚Ì”
		ActivationFunctionType activationFunctionType,			// ’†ŠÔ‘w‚ÌŠˆ«‰»ŠÖ”
		OutputActivationFunctionType outputActivationFunctionType		// o—Í‘w‚ÌŠˆ«‰»ŠÖ”
	)
	{
		int* numNeuronPointer = new int[numNeuron->Length];
		for (int i = 0; i < numNeuron->Length; i++) numNeuronPointer[i] = numNeuron[i];

		int activationFunctionInt = -1;
		int outputActivationFunctionInt = -1;

		switch (activationFunctionType)
		{
		case ActivationFunctionType::Sigmoid:
			activationFunctionInt = FUNCTION_SIGMOID;
			break;
		case ActivationFunctionType::ReLU:
			activationFunctionInt = FUNCTION_RELU;
			break;
		}

		switch (outputActivationFunctionType)
		{
		case OutputActivationFunctionType::None:
			outputActivationFunctionInt = FUNCTION_NONE;
			break;
		case OutputActivationFunctionType::SoftMax:
			outputActivationFunctionInt = FUNCTION_SOFTMAX;
			break;
		case OutputActivationFunctionType::Sigmoid:
			outputActivationFunctionInt = FUNCTION_SIGMOID;
			break;
		case OutputActivationFunctionType::ReLU:
			outputActivationFunctionInt = FUNCTION_RELU;
			break;
		}

		_multiPerceptronCore = new MultiLayerPerceptronCore(
			numInput,
			numNeuron->Length,
			numNeuronPointer,
			activationFunctionInt,
			outputActivationFunctionInt
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
