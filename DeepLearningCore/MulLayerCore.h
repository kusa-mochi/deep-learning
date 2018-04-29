#pragma once
#include "ILayerCore.h"

namespace DeepLearningCore
{
	class MulLayerCore :
		public ILayerCore
	{
	public:
		MulLayerCore();
		virtual ~MulLayerCore();
		virtual void Initialize();
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		LayerBackwardOutput BackwardTwoWay(MatrixXX dout);
	private:
		MatrixXX _x;
		MatrixXX _y;
	};
}
