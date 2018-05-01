#pragma once
#include "Layer.h"

namespace DeepLearningCore
{
	class AffineLayerCore :
		public ILayerCore
	{
	public:
		AffineLayerCore(MatrixXX w, VectorXX b);
		virtual ~AffineLayerCore();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		virtual LayerBackwardOutput Backward(MatrixXX dout);
	private:
		int _numInput = 1;
		MatrixXX _w;
		VectorXX _b;
		MatrixXX _x;
		MatrixXX _dw;
		VectorXX _db;
	};
}
