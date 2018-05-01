#pragma once
#include "Layer.h"

namespace DeepLearningCore
{
	class AddLayerCore :
		public ILayerCore
	{
	public:
		AddLayerCore();
		virtual ~AddLayerCore();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		virtual LayerBackwardOutput Backward(MatrixXX dout);
	private:
		int _numInput = 2;
	};
}
