#include "stdafx.h"
#include "MultiLayerPerceptronCore.h"


namespace DeepLearningCore
{
	WEIGHT_TYPE Through(WEIGHT_TYPE x)
	{
		return x;
	}

	WEIGHT_TYPE Sigmoid(WEIGHT_TYPE x)
	{
		return (1.0 / (1.0 + std::exp(-x)));
	}

	WEIGHT_TYPE ReLU(WEIGHT_TYPE x)
	{
		return (x > 0.0 ? x : 0.0);
	}

	MultiLayerPerceptronCore::MultiLayerPerceptronCore(
		int numInput,
		int numLayer,
		int* numNeuron,
		int activationFunctionType,
		int outputActivationFunction
	)
	{
		if (
			numInput < 1 ||
			numLayer < 2 ||
			activationFunctionType < 0 ||
			outputActivationFunction < 0
			)
		{
			throw ARGUMENT_EXCEPTION;
		}
		if (numNeuron == NULL)
		{
			throw ARGUMENT_NULL_EXCEPTION;
		}

		_numInput = numInput;
		_numLayer = numLayer;
		_numNeuron = numNeuron;

		switch (activationFunctionType)
		{
		case FUNCTION_SIGMOID:
			_ActivationFunction = &Sigmoid;
			break;
		case FUNCTION_RELU:
			_ActivationFunction = &ReLU;
			break;
		default:
			break;
		}

		switch (outputActivationFunction)
		{
		case FUNCTION_NONE:
			_OutputActivationFunction = &Through;
			break;
		case FUNCTION_SIGMOID:
			_OutputActivationFunction = &Sigmoid;
			break;
		case FUNCTION_RELU:
			_OutputActivationFunction = &ReLU;
			break;
		case FUNCTION_SOFTMAX:
			_OutputActivationFunction = NULL;
		default:
			break;
		}

		this->InitializeWeights();
	}


	MultiLayerPerceptronCore::~MultiLayerPerceptronCore()
	{
		delete[] _numNeuron;
		delete[] _weight;
	}


	void MultiLayerPerceptronCore::InitializeWeights()
	{
		_weight = new MatrixXX[_numLayer];
		_weight[0] = MatrixXX::Random(_numInput, _numNeuron[0]);
		for (int iLayer = 1; iLayer < _numLayer; iLayer++)
			_weight[iLayer] = MatrixXX::Random(_numNeuron[iLayer - 1], _numNeuron[iLayer]);
	}


	void MultiLayerPerceptronCore::Predict(WEIGHT_TYPE** input, int numData, WEIGHT_TYPE** output)
	{
		if (input == NULL) { throw ARGUMENT_NULL_EXCEPTION; }
		if (numData < 1) { throw ARGUMENT_EXCEPTION; }
		if (output != NULL) { throw INVALID_OPERATION_EXCEPTION; }

		MatrixXX inputMatrix = this->Pointer2Matrix(input, numData, _numInput);
		MatrixXX tmpMatrix;
		tmpMatrix = (inputMatrix * _weight[0]).unaryExpr(_ActivationFunction);
		for (int iLayer = 1; iLayer < _numLayer - 1; iLayer++)
			tmpMatrix = (tmpMatrix * _weight[iLayer]).unaryExpr(_ActivationFunction);
		MatrixXX outputMatrix;
		if (_OutputActivationFunction == NULL)
		{
			// TODO: Apply Softmax function
		}
		else
		{
			outputMatrix = (tmpMatrix * _weight[_numLayer - 1]).unaryExpr(_OutputActivationFunction);
		}
		output = this->Matrix2Pointer(outputMatrix);
	}

	MatrixXX MultiLayerPerceptronCore::Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols)
	{
		MatrixXX output = MatrixXX::Zero(rows, cols);
		for (int iRow = 0; iRow < rows; iRow++)
			for (int iColumn = 0; iColumn < cols; iColumn++)
				output(iRow, iColumn) = p[iRow][iColumn];
		return output;
	}

	WEIGHT_TYPE** MultiLayerPerceptronCore::Matrix2Pointer(MatrixXX m)
	{
		int rows = m.rows();
		int cols = m.cols();
		WEIGHT_TYPE** p = new WEIGHT_TYPE*[rows];
		for (int iRow = 0; iRow < rows; iRow++)
			p[iRow] = new WEIGHT_TYPE[cols];
		return p;
	}

	//// DLL参照元が呼び出すための関数
	//MultiLayerPerceptronCore* CreateInstance(
	//	int numInput,
	//	int numLayer,
	//	int* numNeuron,
	//	WEIGHT_TYPE(*ActivationFunction)(WEIGHT_TYPE),
	//	WEIGHT_TYPE(*OutputActivationFunction)(WEIGHT_TYPE)
	//)
	//{
	//	return new MultiLayerPerceptronCore(
	//		numInput,
	//		numLayer,
	//		numNeuron,
	//		ActivationFunction,
	//		OutputActivationFunction
	//	);
	//}
	//// インスタンスを破棄
	//void ReleseInstance(MultiLayerPerceptronCore* p)
	//{
	//	delete p;
	//}
}
