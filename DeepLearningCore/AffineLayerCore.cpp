#include "stdafx.h"
#include "AffineLayerCore.h"


namespace DeepLearningCore
{
	AffineLayerCore::AffineLayerCore(MatrixXX* pW, VectorXX* pB)
	{
		_pW = pW;
		_pB = pB;
	}


	AffineLayerCore::~AffineLayerCore()
	{
	}


	MatrixXX AffineLayerCore::Forward(MatrixXX x)
	{
		_x = x;

		MatrixXX output = x * (*_pW);
		for (int iRow = 0; iRow < output.rows(); iRow++)
		{
			for (int iColumn = 0; iColumn < output.cols(); iColumn++)
			{
				output(iRow, iColumn) += (*_pB)(0, iColumn);
			}
		}

		return output;
	}


	MatrixXX AffineLayerCore::Forward(MatrixXX x, MatrixXX y)
	{
		throw NOT_IMPLEMENTED_EXCEPTION;
	}


	LayerBackwardOutput AffineLayerCore::Backward(MatrixXX dout)
	{
		LayerBackwardOutput output;
		output.x = dout * _pW->transpose();
		_dw = _x.transpose() * dout;
		_db = dout.colwise().sum();

		return output;
	}
}
