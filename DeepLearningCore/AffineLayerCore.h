#pragma once

#include "ILayerCore.h"

namespace DeepLearningCore
{
	class AffineLayerCore
	{
	public:
		AffineLayerCore();
		virtual ~AffineLayerCore();
		virtual void Initialize(VectorXX b);
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX BackwardOneWay(MatrixXX dout);
	private:
		MatrixXX _w;
		VectorXX _b;
		MatrixXX _x;
		MatrixXX _dw;
		VectorXX _db;
	};
}
