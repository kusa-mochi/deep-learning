#include "stdafx.h"
#include "AffineLayerCore.h"


namespace DeepLearningCore
{
	AffineLayerCore::AffineLayerCore(MatrixXX w, VectorXX b)
	{
		_w = w;
		_b = b;
	}


	AffineLayerCore::~AffineLayerCore()
	{
	}


	MatrixXX AffineLayerCore::Forward(MatrixXX x)
	{
		_x = x;
		return (x * _w) + _b;
	}


	MatrixXX AffineLayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		throw NOT_IMPLEMENTED_EXCEPTION;
	}


	LayerBackwardOutput AffineLayerCore::Backward(MatrixXX dout)
	{
		LayerBackwardOutput output;
		output.x = dout * _w.transpose();
		_dw = _x.transpose() * dout;
		_db = dout.colwise().sum();

		return output;
	}
}
