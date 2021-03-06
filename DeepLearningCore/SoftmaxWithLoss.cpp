#include "stdafx.h"
#include "SoftmaxWithLoss.h"


namespace DeepLearningCore
{
	SoftmaxWithLoss::SoftmaxWithLoss()
	{
	}


	SoftmaxWithLoss::~SoftmaxWithLoss()
	{
	}


	MatrixXX SoftmaxWithLoss::Forward(MatrixXX x)
	{
		throw NOT_IMPLEMENTED_EXCEPTION;
	}


	MatrixXX SoftmaxWithLoss::Forward(MatrixXX x, MatrixXX t)
	{
		_t = t;
		_y = this->Softmax(x);
		_loss = this->CrossEntrypyError(_y, _t);

		return _loss;
	}


	LayerBackwardOutput SoftmaxWithLoss::Backward(MatrixXX dout = MatrixXX::Ones(1, 1))
	{
		LayerBackwardOutput output;

		// TODO: 誤ってない？？テストケースが合格にならない原因はここ？
		output.x = (_y - _t).array() / _t.rows();

		return output;
	}


	MatrixXX SoftmaxWithLoss::Softmax(MatrixXX x)
	{
		// 各行の最大値
		VectorXX maxValues = x.rowwise().maxCoeff();

		// 指数関数の計算のために値が調整された行列x
		MatrixXX adjustedX = x;

		for (int iRow = 0; iRow < adjustedX.rows(); iRow++)
		{
			for (int iColumn = 0; iColumn < adjustedX.cols(); iColumn++)
			{
				adjustedX(iRow, iColumn) -= maxValues(iRow);
			}
		}

		adjustedX = adjustedX.array().exp();

		for (int iRow = 0; iRow < adjustedX.rows(); iRow++)
		{
			WEIGHT_TYPE sum = 0.0;
			for (int iColumn = 0; iColumn < adjustedX.cols(); iColumn++)
			{
				sum += adjustedX(iRow, iColumn);
			}
			for (int iColumn = 0; iColumn < adjustedX.cols(); iColumn++)
			{
				adjustedX(iRow, iColumn) /= sum;
			}
		}

		return adjustedX;
	}


	MatrixXX SoftmaxWithLoss::CrossEntrypyError(MatrixXX y, MatrixXX t)
	{
		MatrixXX output = MatrixXX::Ones(1, 1);
		output *= -(t.array() * y.array().log().array()).sum() / y.rows();
		return output;
	}
}
