#include "stdafx.h"
#include "SigmoidLayerCore.h"


namespace DeepLearningCore
{
	SigmoidLayerCore::SigmoidLayerCore()
	{
	}


	SigmoidLayerCore::~SigmoidLayerCore()
	{
	}


	MatrixXX SigmoidLayerCore::Forward(MatrixXX x)
	{
		_output = x.unaryExpr(&SigmoidLayerCore::Sigmoid);
		return _output;
	}


	MatrixXX SigmoidLayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		throw NOT_IMPLEMENTED_EXCEPTION;
	}


	LayerBackwardOutput SigmoidLayerCore::Backward(MatrixXX dout)
	{
		LayerBackwardOutput output;
		output.x = dout * (MatrixXX::Ones(_output.rows(), _output.cols()) - _output) * _output;
		return output;
	}


	WEIGHT_TYPE SigmoidLayerCore::Sigmoid(WEIGHT_TYPE x)
	{
		return (1.0 / (1.0 + std::exp(-x)));
	}
}
