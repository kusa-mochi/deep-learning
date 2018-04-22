#include "stdafx.h"
#include "MultiLayerPerceptron.h"


namespace DeepLearning
{
	WEIGHT_TYPE Sigmoid(WEIGHT_TYPE x)
	{
		return (1.0 / (1.0 + std::exp(x)));
	}

	MultiLayerPerceptron::MultiLayerPerceptron(
		int numInput,											// ���͂̎�����
		cli::array<int>^ numNeuron,								// �e�w�̃j���[�����̐�
		ActivationFunctionType activationFunctionType,			// ���ԑw�̊������֐�
		OutputActivationFunctionType outputActivationFunctionType		// �o�͑w�̊������֐�
	)
	{
		int* numNeuronPointer = new int[numNeuron->Length];
		for (int i = 0; i < numNeuron->Length; i++) numNeuronPointer[i] = numNeuron[i];

		_multiPerceptronCore = new MultiLayerPerceptronCore(
			numInput,
			numNeuron->Length,
			numNeuronPointer,
			FUNCTION_SIGMOID,	// TODO: �R���X�g���N�^�̓��͂ɂ��l�����肷��B
			FUNCTION_NONE		// TODO: �R���X�g���N�^�̓��͂ɂ��l�����肷��B
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
