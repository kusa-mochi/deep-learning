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

		// TODO: ����ĂȂ��H�H�e�X�g�P�[�X�����i�ɂȂ�Ȃ������͂����H
		output.x = (_y - _t).array() / _t.rows();

		return output;
	}


	MatrixXX SoftmaxWithLoss::Softmax(MatrixXX x)
	{
		// �e�s�̍ő�l
		VectorXX maxValues = x.rowwise().maxCoeff();

		// �w���֐��̌v�Z�̂��߂ɒl���������ꂽ�s��x
		MatrixXX adjustedX = x;

		for (int iRow = 0; iRow < adjustedX.rows(); iRow++)
		{
			for (int iColumn = 0; iColumn < adjustedX.cols(); iColumn++)
			{
				adjustedX(iRow, iColumn) -= maxValues(iRow);
			}
		}

		//adjustedX.rowwise() -= maxValues;

		//MatrixXX adjustedX = MatrixXX::Zero(x.rows(), x.cols());
		//for (int iRow = 0; iRow < x.rows(); iRow++)
		//{
		//	for (int iColumn = 0; iColumn < x.cols(); iColumn++)
		//	{
		//		adjustedX(iRow, iColumn) = x(iRow, iColumn) - maxValues(iRow, 0);
		//	}
		//}
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

		//// �e�s�̍��v�l
		//VectorXX sums = adjustedX.rowwise().sum();

		//// softmax�֐��l���v�Z����B
		//adjustedX.rowwise() -= sums;

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
