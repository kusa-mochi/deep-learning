#include "stdafx.h"
#include "MultiLayerPerceptronCore.h"


namespace DeepLearningCore
{
	MultiLayerPerceptronCore::MultiLayerPerceptronCore(
		int numInput,
		int numLayer,
		LayerInfo* layerInfo
	)
	{
		assert(numInput >= 1);
		assert(numLayer >= 2);
		if (
			numInput < 1 ||
			numLayer < 2
			)
		{
			throw ARGUMENT_EXCEPTION;
		}

		assert(layerInfo != NULL);
		if (
			layerInfo == NULL
			)
		{
			throw ARGUMENT_NULL_EXCEPTION;
		}

		_numInput = numInput;
		_numLayer = numLayer;
		_layerInfo = layerInfo;
		this->InitializeWeights();
		this->InitializeLayers();
		this->InitializeLastLayer();
	}


	MultiLayerPerceptronCore::~MultiLayerPerceptronCore()
	{
		delete[] _weight;
		_weight = NULL;
		delete[] _bias;
		_bias = NULL;
	}


	void MultiLayerPerceptronCore::InitializeWeights()
	{
		_weight = new MatrixXX[_numLayer];
		_weight[0] = MatrixXX::Random(_numInput, _layerInfo[0].NumNeuron);
		for (int iLayer = 1; iLayer < _numLayer; iLayer++)
			_weight[iLayer] = MatrixXX::Random(_layerInfo[iLayer - 1].NumNeuron, _layerInfo[iLayer].NumNeuron);

		_bias = new VectorXX[_numLayer];
		for (int iLayer = 0; iLayer < _numLayer; iLayer++)
			_bias[iLayer] = VectorXX::Random(1, _layerInfo[iLayer].NumNeuron);
	}


	void MultiLayerPerceptronCore::InitializeLayers()
	{
		Layer* prev = NULL;
		Layer* p = NULL;
		for (int iLayer = 0; iLayer < _numLayer; iLayer++)
		{
			p = new Layer;
			if (iLayer == 0)
			{
				_layer = p;
			}
			else
			{
				prev->Next = p;
			}

			p->Layer = new AffineLayerCore(&_weight[iLayer], &_bias[iLayer]);
			p->LayerType = _LayerType::Affine;

			// 出力層の場合
			if (iLayer == _numLayer - 1)
			{
				// 次の層はないことにする。
				p->Next = NULL;

				// この関数の処理を終了する。
				return;
			}

			p->Next = new Layer;
			prev = p;
			p = p->Next;

			switch (_layerInfo[iLayer].LayerType)
			{
			case _LayerType::None:
				p->Layer = new IdentityLayerCore();
				p->LayerType = _LayerType::None;
				break;
			case _LayerType::Affine:
				p->Layer = new AffineLayerCore(&_weight[iLayer], &_bias[iLayer]);
				p->LayerType = _LayerType::Affine;
				break;
			case _LayerType::Sigmoid:
				p->Layer = new SigmoidLayerCore();
				p->LayerType = _LayerType::Sigmoid;
				break;
			case _LayerType::ReLU:
				p->Layer = new ReLULayerCore();
				p->LayerType = _LayerType::ReLU;
				break;
			default:
				throw INVALID_OPERATION_EXCEPTION;
			}

			p->Next = NULL;
			prev = p;
			p = p->Next;
		}
	}


	void MultiLayerPerceptronCore::InitializeLastLayer()
	{
		switch (_layerInfo[_numLayer - 1].LayerType)
		{
		case _LayerType::None:
			_lastLayer->Layer = new IdentityLayerCore();
			_lastLayer->LayerType = _LayerType::None;
			break;
		case _LayerType::Affine:
			_lastLayer->Layer = new AffineLayerCore(&_weight[_numLayer - 1], &_bias[_numLayer - 1]);
			_lastLayer->LayerType = _LayerType::Affine;
			break;
		case _LayerType::Sigmoid:
			_lastLayer->Layer = new SigmoidLayerCore();
			_lastLayer->LayerType = _LayerType::Sigmoid;
			break;
		case _LayerType::ReLU:
			_lastLayer->Layer = new ReLULayerCore();
			_lastLayer->LayerType = _LayerType::ReLU;
			break;
		case _LayerType::SoftMax:
			_lastLayer->Layer = new SoftmaxWithLoss();
			_lastLayer->LayerType = _LayerType::SoftMax;
			break;
		default:
			throw INVALID_OPERATION_EXCEPTION;
		}
	}


	void MultiLayerPerceptronCore::SetWeights(WEIGHT_TYPE*** weights)
	{
		assert(weights != NULL);
		assert(weights[_numLayer - 1] != NULL);

		if (weights == NULL) { throw ARGUMENT_NULL_EXCEPTION; }

		for (int iNeuron = 0; iNeuron < _layerInfo[0].NumNeuron; iNeuron++)
		{
			for (int iInput = 0; iInput < _numInput; iInput++)
			{
				_weight[0](iInput, iNeuron) = weights[0][iInput][iNeuron];
			}
		}
		for (int iLayer = 1; iLayer < _numLayer; iLayer++)
		{
			for (int iNeuron = 0; iNeuron < _layerInfo[iLayer].NumNeuron; iNeuron++)
			{
				for (int iInput = 0; iInput < _layerInfo[iLayer - 1].NumNeuron; iInput++)
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

		for (int iLayer = 0; iLayer < _numLayer; iLayer++)
		{
			for (int iNeuron = 0; iNeuron < _layerInfo[iLayer].NumNeuron; iNeuron++)
			{
				_bias[iLayer](iNeuron) = bias[iLayer][iNeuron];
			}
		}
	}


	void MultiLayerPerceptronCore::Predict(WEIGHT_TYPE** input, int numData, WEIGHT_TYPE*** output)
	{
		assert(input != NULL);
		assert(input[_numInput - 1] != NULL);
		assert(numData >= 1);
		assert(*output == NULL);
		if (input == NULL) { throw ARGUMENT_NULL_EXCEPTION; }
		if (numData < 1) { throw ARGUMENT_EXCEPTION; }
		if (*output != NULL) { throw INVALID_OPERATION_EXCEPTION; }

		MatrixXX tmpMatrix = this->Pointer2Matrix(input, numData, _numInput);
		tmpMatrix = this->PredictCore(tmpMatrix);
		this->Matrix2Pointer(tmpMatrix, output);
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

		int rows = (int)m.rows();
		int cols = (int)m.cols();
		*p = new WEIGHT_TYPE*[rows];
		for (int iRow = 0; iRow < rows; iRow++)
		{
			(*p)[iRow] = new WEIGHT_TYPE[cols];
			for (int iColumn = 0; iColumn < cols; iColumn++)
			{
				(*p)[iRow][iColumn] = m(iRow, iColumn);
			}
		}
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


	MatrixXX MultiLayerPerceptronCore::PredictCore(MatrixXX input)
	{
		for (Layer* pLayer = _layer; pLayer != NULL; pLayer = pLayer->Next)
		{
			input = pLayer->Layer->Forward(input);
		}

		MatrixXX output = input;

		return output;
	}


	MatrixXX MultiLayerPerceptronCore::ApplyLastLayer(MatrixXX m, MatrixXX t)
	{
		return _lastLayer->Layer->Forward(m, t);
	}


	MatrixXX MultiLayerPerceptronCore::Loss(MatrixXX m, MatrixXX t)
	{
		MatrixXX y = this->PredictCore(m);
		MatrixXX output = this->ApplyLastLayer(y, t);
		return output;
	}


	MatrixXX* MultiLayerPerceptronCore::Gradient(MatrixXX m, MatrixXX t)
	{
		this->Loss(m, t);

		// TODO

		// TODO
		return _weight;
	}
}
