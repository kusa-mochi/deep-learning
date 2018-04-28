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
		int outputActivationFunctionType
	)
	{
		//assert(numInput >= 1);
		//assert(numLayer >= 2);
		//assert(activationFunctionType >= 0);
		//assert(outputActivationFunctionType >= 0);
		//if (
		//	numInput < 1 ||
		//	numLayer < 2 ||
		//	activationFunctionType < 0 ||
		//	outputActivationFunctionType < 0
		//	)
		//{
		//	throw ARGUMENT_EXCEPTION;
		//}

		//assert(numNeuron != NULL);
		//if (numNeuron == NULL)
		//{
		//	throw ARGUMENT_NULL_EXCEPTION;
		//}

		//_numInput = numInput;
		//_numLayer = numLayer;
		//_numNeuron = new int[numLayer];
		//for (int iLayer = 0; iLayer < numLayer; iLayer++)
		//{
		//	_numNeuron[iLayer] = numNeuron[iLayer];
		//}

		//switch (activationFunctionType)
		//{
		//case FUNCTION_SIGMOID:
		//	_ActivationFunction = &Sigmoid;
		//	break;
		//case FUNCTION_RELU:
		//	_ActivationFunction = &ReLU;
		//	break;
		//default:
		//	break;
		//}

		//switch (outputActivationFunctionType)
		//{
		//case FUNCTION_NONE:
		//	_OutputActivationFunction = &Through;
		//	break;
		//case FUNCTION_SIGMOID:
		//	_OutputActivationFunction = &Sigmoid;
		//	break;
		//case FUNCTION_RELU:
		//	_OutputActivationFunction = &ReLU;
		//	break;
		//case FUNCTION_SOFTMAX:
		//	_OutputActivationFunction = NULL;
		//default:
		//	break;
		//}

		//this->InitializeWeights();
	}


	MultiLayerPerceptronCore::~MultiLayerPerceptronCore()
	{
		//delete[] _numNeuron;
		//_numNeuron = NULL;
		//delete[] _weight;
		//_weight = NULL;
		//delete[] _bias;
		//_bias = NULL;
	}


	void MultiLayerPerceptronCore::InitializeWeights()
	{
		_weight = new MatrixXX[_numLayer];
		_weight[0] = MatrixXX::Random(_numInput, _numNeuron[0]);
		for (int iLayer = 1; iLayer < _numLayer; iLayer++)
			_weight[iLayer] = MatrixXX::Random(_numNeuron[iLayer - 1], _numNeuron[iLayer]);

		_bias = new VectorXX[_numLayer];
		for (int iLayer = 0; iLayer < _numLayer; iLayer++)
			_bias[iLayer] = VectorXX::Random(1, _numNeuron[iLayer]);
	}


	void MultiLayerPerceptronCore::SetWeights(WEIGHT_TYPE*** weights)
	{
		assert(weights != NULL);
		assert(weights[_numLayer - 1] != NULL);

		if (weights == NULL) { throw ARGUMENT_NULL_EXCEPTION; }

		for (int iNeuron = 0; iNeuron < _numNeuron[0]; iNeuron++)
		{
			for (int iInput = 0; iInput < _numInput; iInput++)
			{
				_weight[0](iInput, iNeuron) = weights[0][iInput][iNeuron];
			}
		}
		for (int iLayer = 1; iLayer < _numLayer; iLayer++)
		{
			for (int iNeuron = 0; iNeuron < _numNeuron[iLayer]; iNeuron++)
			{
				for (int iInput = 0; iInput < _numNeuron[iLayer - 1]; iInput++)
				{
					_weight[iLayer](iInput, iNeuron) = weights[iLayer][iInput][iNeuron];
				}
			}
		}
	}


	void MultiLayerPerceptronCore::SetBias(WEIGHT_TYPE** bias)
	{
		assert(bias != NULL);
		assert(bias[_numLayer - 1] != NULL);

		if (bias == NULL) { throw ARGUMENT_NULL_EXCEPTION; }

		for (int iLayer = 1; iLayer < _numLayer; iLayer++)
		{
			for (int iNeuron = 0; iNeuron < _numNeuron[iLayer]; iNeuron++)
			{
				_bias[iLayer](iNeuron) = bias[iLayer][iNeuron];
			}
		}
	}


	void MultiLayerPerceptronCore::Predict(WEIGHT_TYPE** input, int numData, WEIGHT_TYPE** output)
	{
		assert(input != NULL);
		assert(input[_numInput - 1] != NULL);
		assert(numData >= 1);
		assert(output == NULL);
		if (input == NULL) { throw ARGUMENT_NULL_EXCEPTION; }
		if (numData < 1) { throw ARGUMENT_EXCEPTION; }
		if (output != NULL) { throw INVALID_OPERATION_EXCEPTION; }

		assert(7 == 8);

		MatrixXX inputMatrix = this->Pointer2Matrix(input, numData, _numInput);
		MatrixXX tmpMatrix;
		assert(5 == 6);
		tmpMatrix = inputMatrix * _weight[0];
		assert(3 == 4);
		tmpMatrix = this->MatrixPlusVector(tmpMatrix, _bias[0]);
		assert(1 == 2);
		tmpMatrix = tmpMatrix.unaryExpr(_ActivationFunction);
		assert(1 == 2);
		for (int iLayer = 1; iLayer < _numLayer - 1; iLayer++)
		{
			tmpMatrix = tmpMatrix * _weight[iLayer];
			tmpMatrix = this->MatrixPlusVector(tmpMatrix, _bias[iLayer]);
			tmpMatrix = tmpMatrix.unaryExpr(_ActivationFunction);
		}
		MatrixXX outputMatrix;
		if (_OutputActivationFunction == NULL)
		{
			// TODO: Apply Softmax function
		}
		else
		{
			tmpMatrix = tmpMatrix * _weight[_numLayer - 1];
			tmpMatrix = this->MatrixPlusVector(tmpMatrix, _bias[_numLayer - 1]);
			outputMatrix = tmpMatrix.unaryExpr(_OutputActivationFunction);
		}
		this->Matrix2Pointer(outputMatrix, &output);
	}

	MatrixXX MultiLayerPerceptronCore::Pointer2Matrix(WEIGHT_TYPE** p, int rows, int cols)
	{
		assert(p != NULL);
		assert(rows >= 1);
		assert(cols >= 1);

		MatrixXX output = MatrixXX::Zero(rows, cols);
		for (int iRow = 0; iRow < rows; iRow++)
			for (int iColumn = 0; iColumn < cols; iColumn++)
				output(iRow, iColumn) = p[iRow][iColumn];
		return output;
	}

	void MultiLayerPerceptronCore::Matrix2Pointer(MatrixXX m, WEIGHT_TYPE*** p)
	{
		assert(m.rows() >= 1);
		assert(m.cols() >= 1);

		int rows = m.rows();
		int cols = m.cols();
		*p = new WEIGHT_TYPE*[rows];
		for (int iRow = 0; iRow < rows; iRow++)
			*p[iRow] = new WEIGHT_TYPE[cols];
	}

	MatrixXX MultiLayerPerceptronCore::MatrixPlusVector(MatrixXX m, VectorXX v)
	{
		assert(m.rows() >= 1);
		assert(m.cols() >= 1);
		assert(v.rows() == 1);
		assert(v.cols() >= 1);
		assert(m.cols() == v.cols());

		MatrixXX output = MatrixXX::Zero(m.rows(), m.cols());

		for (int iRow = 0; iRow < m.rows(); iRow++)
			for (int iColumn = 0; iColumn < m.cols(); iColumn++)
				output(iRow, iColumn) = m(iRow, iColumn) + v(iColumn);

		return output;
	}
}
