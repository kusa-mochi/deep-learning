#include "stdafx.h"
#include "IdentityLayerCore.h"


namespace DeepLearningCore
{
	IdentityLayerCore::IdentityLayerCore()
	{
	}


	IdentityLayerCore::~IdentityLayerCore()
	{
	}


	MatrixXX IdentityLayerCore::Forward(MatrixXX x)
	{
		return x;
	}


	MatrixXX IdentityLayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		throw NOT_IMPLEMENTED_EXCEPTION;
	}


	LayerBackwardOutput IdentityLayerCore::Backward(MatrixXX dout)
	{
		LayerBackwardOutput output;
		output.x = dout;
		return output;
	}
}
