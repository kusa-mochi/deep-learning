#pragma once
#include "Layer.h"

namespace DeepLearningCore
{
	class IdentityLayerCore :
		public ILayerCore
	{
	public:
		IdentityLayerCore();
		virtual ~IdentityLayerCore();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		virtual LayerBackwardOutput Backward(MatrixXX dout);
	private:
		int _numInput = 1;
	};
}
