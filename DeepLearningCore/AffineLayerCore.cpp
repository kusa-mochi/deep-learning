#include "stdafx.h"
#include "AffineLayerCore.h"


namespace DeepLearningCore
{
	AffineLayerCore::AffineLayerCore(MatrixXX* pW, VectorXX* pB)
	{
		assert((*pW).rows() > 0);
		assert((*pW).cols() > 0);
		assert((*pB).rows() > 0);
		assert((*pB).cols() > 0);
		assert((*pW).cols() == (*pB).cols());
		assert((*pB).rows() == 1);

		_pW = pW;
		_pB = pB;
	}


	AffineLayerCore::~AffineLayerCore()
	{
	}


	MatrixXX AffineLayerCore::Forward(MatrixXX x)
	{
		assert(x.rows() > 0);
		assert(x.cols() > 0);
		assert(x.cols() == (*_pW).rows());

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
		assert(dout.rows() > 0);
		assert(dout.cols() > 0);
		assert(dout.cols() == (*_pW).cols());
		assert(_x.rows() == dout.rows());

		LayerBackwardOutput output;
		output.x = dout * _pW->transpose();
		_dw = _x.transpose() * dout;
		_db = dout.colwise().sum();

		assert(_db.rows() == 1);
		assert(_db.cols() == dout.cols());

		return output;
	}
}
