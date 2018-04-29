#pragma once

#include "DeepLearningCore.h"

namespace DeepLearningCore
{
	class ILayerCore
	{
	public:
		virtual ~ILayerCore() {}
		virtual void Initialize() = 0;
		virtual void Initialize(VectorXX b) = 0;
		virtual MatrixXX Forward(MatrixXX x) = 0;
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y) = 0;
		virtual MatrixXX BackwardOneWay(MatrixXX dout) = 0;
		virtual LayerBackwardOutput BackwardTwoWay(MatrixXX dout) = 0;
	private:
		MatrixXX _output;
	};
}
