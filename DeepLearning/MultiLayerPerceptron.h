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
			int numInput,											// “ü—Í‚ÌŸŒ³”
			cli::array<int>^ numNeuron,								// Še‘w‚Ìƒjƒ…[ƒƒ“‚Ì”
			ActivationFunctionType activationFunctionType,			// ’†ŠÔ‘w‚ÌŠˆ«‰»ŠÖ”
			OutputActivationFunctionType outputActivationFunctionType	// o—Í‘w‚ÌŠˆ«‰»ŠÖ”
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
