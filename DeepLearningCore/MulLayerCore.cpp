#include "stdafx.h"
#include "MulLayerCore.h"


namespace DeepLearningCore
{
	MulLayerCore::MulLayerCore()
	{
		this->Initialize();
	}


	MulLayerCore::~MulLayerCore()
	{
	}


	void MulLayerCore::Initialize()
	{

	}


	MatrixXX MulLayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		_x = x;
		_y = y;
		return x * y;
	}


	LayerBackwardOutput MulLayerCore::BackwardTwoWay(MatrixXX dout)
	{
		LayerBackwardOutput output;

		output.x = dout * _y;
		output.y = dout * _x;

		return output;
	}
}
