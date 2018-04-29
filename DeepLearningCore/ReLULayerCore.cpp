#include "stdafx.h"
#include "ReLULayerCore.h"


namespace DeepLearningCore
{
	ReLULayerCore::ReLULayerCore()
	{
		this->Initialize();
	}


	ReLULayerCore::~ReLULayerCore()
	{
	}


	void ReLULayerCore::Initialize()
	{

	}


	MatrixXX ReLULayerCore::Forward(MatrixXX x)
	{
		MatrixXX output = x.unaryExpr(&ReLULayerCore::ReLU);
		_output = x.unaryExpr(&ReLULayerCore::ReLUBool);
		return output;
	}


	MatrixXX ReLULayerCore::BackwardOneWay(MatrixXX dout)
	{
		return _output.select(dout, MatrixXX::Zero(dout.size()));
	}


	WEIGHT_TYPE ReLULayerCore::ReLU(WEIGHT_TYPE x)
	{
		return (x > 0.0 ? x : 0.0);
	}


	bool ReLULayerCore::ReLUBool(WEIGHT_TYPE x)
	{
		return x > 0.0;
	}
}
