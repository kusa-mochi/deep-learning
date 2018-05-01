#include "stdafx.h"
#include "ReLULayerCore.h"


namespace DeepLearningCore
{
	ReLULayerCore::ReLULayerCore()
	{
	}


	ReLULayerCore::~ReLULayerCore()
	{
	}


	MatrixXX ReLULayerCore::Forward(MatrixXX x)
	{
		MatrixXX output = x.unaryExpr(&ReLULayerCore::ReLU);
		_output = x.unaryExpr(&ReLULayerCore::ReLUBool).cast<bool>();
		return output;
	}


	MatrixXX ReLULayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		throw NOT_IMPLEMENTED_EXCEPTION;
	}


	LayerBackwardOutput ReLULayerCore::Backward(MatrixXX dout)
	{
		LayerBackwardOutput output;
		output.x = _output.select(dout, MatrixXX::Zero(dout.rows(), dout.cols()));
		return output;
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
