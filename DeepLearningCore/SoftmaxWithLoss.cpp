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

		output.x = (_y - _t).array() / _t.rows();

		return output;
	}


	MatrixXX SoftmaxWithLoss::Softmax(MatrixXX x)
	{
		// 各行の最大値
		VectorXX maxValues = x.rowwise().maxCoeff();

		// 指数関数の計算のために値が調整された行列x
		MatrixXX adjustedX = x;
		adjustedX.rowwise() -= maxValues;
		//MatrixXX adjustedX = MatrixXX::Zero(x.rows(), x.cols());
		//for (int iRow = 0; iRow < x.rows(); iRow++)
		//{
		//	for (int iColumn = 0; iColumn < x.cols(); iColumn++)
		//	{
		//		adjustedX(iRow, iColumn) = x(iRow, iColumn) - maxValues(iRow, 0);
		//	}
		//}
		adjustedX = adjustedX.array().exp();

		// 各行の合計値
		VectorXX sums = adjustedX.rowwise().sum();

		// softmax関数値を計算する。
		adjustedX.rowwise() -= sums;
		//for (int iRow = 0; iRow < x.rows(); iRow++)
		//{
		//	for (int iColumn = 0; iColumn < x.cols(); iColumn++)
		//	{
		//		adjustedX(iRow, iColumn) = adjustedX(iRow, iColumn) - sums(iRow, 0);
		//	}
		//}

		return adjustedX;
	}


	MatrixXX SoftmaxWithLoss::CrossEntrypyError(MatrixXX y, MatrixXX t)
	{
		MatrixXX output = MatrixXX::Ones(1, 1);
		output *= -(t.array() * y.array().log().array()).sum() / y.cols();
		return output;
	}
}
