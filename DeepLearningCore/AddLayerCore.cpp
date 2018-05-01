#include "stdafx.h"
#include "AddLayerCore.h"


namespace DeepLearningCore
{
	AddLayerCore::AddLayerCore()
	{
	}


	AddLayerCore::~AddLayerCore()
	{
	}


	MatrixXX AddLayerCore::Forward(MatrixXX x)
	{
		throw NOT_IMPLEMENTED_EXCEPTION;
	}


	MatrixXX AddLayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		return (x + y);
	}


	LayerBackwardOutput AddLayerCore::Backward(MatrixXX dout)
	{
		LayerBackwardOutput output;
		output.x = dout;
		output.y = dout;

		return output;
	}
}
