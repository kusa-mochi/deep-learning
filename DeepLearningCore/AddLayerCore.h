#pragma once

#include "ILayerCore.h"

namespace DeepLearningCore
{
	class AddLayerCore
		: ILayerCore
	{
	public:
		AddLayerCore();
		virtual ~AddLayerCore();
		virtual void Initialize();
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		virtual LayerBackwardOutput BackwardTwoWay(MatrixXX dout);
	};
}
