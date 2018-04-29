#pragma once
#include "ILayerCore.h"

namespace DeepLearningCore
{
	class ReLULayerCore :
		public ILayerCore
	{
	public:
		ReLULayerCore();
		virtual ~ReLULayerCore();
		virtual void Initialize();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX BackwardOneWay(MatrixXX dout);
	private:
		Matrix<bool, Dynamic, Dynamic> _output;
		static WEIGHT_TYPE ReLU(WEIGHT_TYPE x);
		static bool ReLUBool(WEIGHT_TYPE x);
	};
}
