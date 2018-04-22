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
			int numInput,											// ���͂̎�����
			cli::array<int>^ numNeuron,								// �e�w�̃j���[�����̐�
			ActivationFunctionType activationFunctionType,			// ���ԑw�̊������֐�
			OutputActivationFunctionType outputActivationFunctionType	// �o�͑w�̊������֐�
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
