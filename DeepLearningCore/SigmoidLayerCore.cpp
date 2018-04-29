#include "stdafx.h"
#include "SigmoidLayerCore.h"


namespace DeepLearningCore
{
	SigmoidLayerCore::SigmoidLayerCore()
	{
		this->Initialize();
	}


	SigmoidLayerCore::~SigmoidLayerCore()
	{
	}


	void SigmoidLayerCore::Initialize()
	{

	}


	MatrixXX SigmoidLayerCore::Forward(MatrixXX x)
	{
		_output = x.unaryExpr(&SigmoidLayerCore::Sigmoid);
		return _output;
	}


	MatrixXX SigmoidLayerCore::BackwardOneWay(MatrixXX dout)
	{
		return (dout * (MatrixXX::Ones(_output.size()) - _output) * _output);
	}


	WEIGHT_TYPE SigmoidLayerCore::Sigmoid(WEIGHT_TYPE x)
	{
		return (1.0 / (1.0 + std::exp(-x)));
	}
}
