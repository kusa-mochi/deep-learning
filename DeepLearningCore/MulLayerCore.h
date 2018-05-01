#pragma once
#include "Layer.h"

namespace DeepLearningCore
{
	class MulLayerCore :
		public ILayerCore
	{
	public:
		MulLayerCore();
		virtual ~MulLayerCore();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		LayerBackwardOutput Backward(MatrixXX dout);
	private:
		int _numInput = 2;
		MatrixXX _x;
		MatrixXX _y;
	};
}
