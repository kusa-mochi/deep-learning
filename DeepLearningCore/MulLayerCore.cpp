#include "stdafx.h"
#include "MulLayerCore.h"


namespace DeepLearningCore
{
	MulLayerCore::MulLayerCore()
	{
	}


	MulLayerCore::~MulLayerCore()
	{
	}


	MatrixXX MulLayerCore::Forward(MatrixXX x)
	{
		throw NOT_IMPLEMENTED_EXCEPTION;
	}


	MatrixXX MulLayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		_x = x;
		_y = y;
		return x * y;
	}


	LayerBackwardOutput MulLayerCore::Backward(MatrixXX dout)
	{
		LayerBackwardOutput output;

		output.x = dout * _y;
		output.y = dout * _x;

		return output;
	}
}
