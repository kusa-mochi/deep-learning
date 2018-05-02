#pragma once
#include "Layer.h"

namespace DeepLearningCore
{
	class SoftmaxWithLoss :
		public ILayerCore
	{
	public:
		SoftmaxWithLoss();
		virtual ~SoftmaxWithLoss();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX Forward(MatrixXX x, MatrixXX t);
		virtual LayerBackwardOutput Backward(MatrixXX dout);
	private:
		MatrixXX _y;
		MatrixXX _t;
		MatrixXX _loss;
		MatrixXX Softmax(MatrixXX x);
		MatrixXX CrossEntrypyError(MatrixXX y, MatrixXX t);
	};
}
