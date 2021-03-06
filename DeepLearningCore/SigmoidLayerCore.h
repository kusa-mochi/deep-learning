#pragma once
#include "Layer.h"

namespace DeepLearningCore
{
	class SigmoidLayerCore :
		public ILayerCore
	{
	public:
		SigmoidLayerCore();
		virtual ~SigmoidLayerCore();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		virtual LayerBackwardOutput Backward(MatrixXX dout);
		static WEIGHT_TYPE Sigmoid(WEIGHT_TYPE x);
	private:
		int _numInput = 1;
		MatrixXX _output;
	};
}
