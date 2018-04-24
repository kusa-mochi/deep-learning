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
			int numInput,											// ���͂̎�����
			cli::array<int>^ numNeuron,								// �e�w�̃j���[�����̐�
			ActivationFunctionType activationFunctionType,			// ���ԑw�̊������֐�
			OutputActivationFunctionType outputActivationFunctionType	// �o�͑w�̊������֐�
		);
		virtual ~MultiLayerPerceptron();
		void SetWeights(
			cli::array<WEIGHT_TYPE, 3>^ weights
		);
		cli::array<WEIGHT_TYPE, 2>^ Predict(
			cli::array<WEIGHT_TYPE, 2>^ input
		);
	private:
		int _numInput = 0;
		MultiLayerPerceptronCore * _multiLayerPerceptronCore = NULL;
		void ManagedArray2NativeArray(cli::array<WEIGHT_TYPE, 2>^ input, WEIGHT_TYPE** output);
		cli::array<WEIGHT_TYPE, 2>^ NativeArray2ManagedArray(WEIGHT_TYPE** input, int rows, int cols);
	};
}
