#pragma once

#include "ILayerCore.h"

namespace DeepLearningCore
{
	class SigmoidLayerCore
	{
	public:
		SigmoidLayerCore();
		virtual ~SigmoidLayerCore();
		virtual void Initialize();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX BackwardOneWay(MatrixXX dout);
		static WEIGHT_TYPE Sigmoid(WEIGHT_TYPE x);
	private:
		MatrixXX _output;
	};
}
