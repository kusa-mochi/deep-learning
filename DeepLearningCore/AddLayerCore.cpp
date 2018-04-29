#include "stdafx.h"
#include "AddLayerCore.h"


namespace DeepLearningCore
{
	AddLayerCore::AddLayerCore()
	{
		this->Initialize();
	}


	AddLayerCore::~AddLayerCore()
	{
	}


	void AddLayerCore::Initialize()
	{

	}


	MatrixXX AddLayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		return (x + y);
	}


	LayerBackwardOutput AddLayerCore::BackwardTwoWay(MatrixXX dout)
	{
		LayerBackwardOutput output;
		output.x = dout;
		output.y = dout;

		return output;
	}
}
