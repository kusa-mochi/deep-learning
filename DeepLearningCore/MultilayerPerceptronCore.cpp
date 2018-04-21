#include "stdafx.h"
#include "MultilayerPerceptronCore.h"


namespace DeepLearningCore
{
	MultilayerPerceptronCore::MultilayerPerceptronCore(
		int numInput,
		vector<int> numNeuron,
		WEIGHT_TYPE(*ActivationFunction)(WEIGHT_TYPE),
		WEIGHT_TYPE(*OutputActivationFunction)(WEIGHT_TYPE)
	)
	{
		if (
			numInput < 1 ||
			numNeuron.size() < 2
			)
		{
			throw ARGUMENT_EXCEPTION;
		}
		if (
			ActivationFunction == NULL ||
			OutputActivationFunction == NULL
			)
		{
			throw ARGUMENT_NULL_EXCEPTION;
		}

		_numLayer = numNeuron.size();
		_numInput = numInput;
		_numNeuron = numNeuron;
		_ActivationFunction = ActivationFunction;
		_OutputActivationFunction = OutputActivationFunction;

		this->InitializeWeights();
	}


	MultilayerPerceptronCore::~MultilayerPerceptronCore()
	{
	}


	void MultilayerPerceptronCore::InitializeWeights()
	{
		_weight = vector<MatrixXd>(_numLayer);
		_weight[0](_numInput, _numNeuron[0]);	// ’†ŠÔ‘w‚Ì‘æˆêŠK‘w‚Ìd‚İ
		for (int iLayer = 1; iLayer < _numLayer; iLayer++)
		{
			_weight[iLayer] = MatrixXd::Random(_numNeuron[iLayer - 1], _numNeuron[iLayer]);
		}
	}


	MatrixXX MultilayerPerceptronCore::Predict(MatrixXX input)
	{
		if (
			input.rows() < 1 ||
			input.cols() < 1
			)
		{
			throw ARGUMENT_EXCEPTION;
		}

		return MatrixXX::Random(1, 1);

		MatrixXX tmpMatrix;
		tmpMatrix = (input * _weight[0]).unaryExpr(_ActivationFunction);
		for (int iLayer = 1; iLayer < _numLayer - 1; iLayer++)
		{
			tmpMatrix = (tmpMatrix * _weight[iLayer]).unaryExpr(_ActivationFunction);
		}
		MatrixXX output = (tmpMatrix * _weight[_numLayer - 1]).unaryExpr(_OutputActivationFunction);
		return output;
	}
}
