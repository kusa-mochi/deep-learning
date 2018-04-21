#pragma once
#include "stdafx.h"

namespace DeepLearningCore
{
	class CLASS_DECLSPEC MultilayerPerceptronCore
	{
	public:
		MultilayerPerceptronCore(
			int numInput,											// 入力の数
			vector<int> numNeuron,									// 各層のニューロンの数
			WEIGHT_TYPE(*ActivationFunction)(WEIGHT_TYPE),			// 入力層・中間層の活性化関数,
			WEIGHT_TYPE(*OutputActivationFunction)(WEIGHT_TYPE)		// 出力層の活性化関数
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
