#pragma once
#include "stdafx.h"

namespace DeepLearningCore
{
	class CLASS_DECLSPEC MultilayerPerceptronCore
	{
	public:
		MultilayerPerceptronCore(
			int numInput,											// ���͂̐�
			vector<int> numNeuron,									// �e�w�̃j���[�����̐�
			WEIGHT_TYPE(*ActivationFunction)(WEIGHT_TYPE),			// ���͑w�E���ԑw�̊������֐�,
			WEIGHT_TYPE(*OutputActivationFunction)(WEIGHT_TYPE)		// �o�͑w�̊������֐�
		);
		virtual ~MultilayerPerceptronCore();
		MatrixXX Predict(MatrixXX input);
	private:
		int _numLayer = 0;
		int _numInput = 0;
		vector<int> _numNeuron;
		WEIGHT_TYPE(*_ActivationFunction)(WEIGHT_TYPE) = NULL;
		WEIGHT_TYPE(*_OutputActivationFunction)(WEIGHT_TYPE) = NULL;
		vector<MatrixXX> _weight;
		void InitializeWeights();
	};
}
