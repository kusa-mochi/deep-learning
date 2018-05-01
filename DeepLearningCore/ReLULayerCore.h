#pragma once
#include "Layer.h"

namespace DeepLearningCore
{
	class ReLULayerCore :
		public ILayerCore
	{
	public:
		ReLULayerCore();
		virtual ~ReLULayerCore();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		virtual LayerBackwardOutput Backward(MatrixXX dout);
	private:
		int _numInput = 1;
		Matrix<bool, Dynamic, Dynamic> _output;
		static WEIGHT_TYPE ReLU(WEIGHT_TYPE x);
		static bool ReLUBool(WEIGHT_TYPE x);
	};
}
