#pragma once
#include "Layer.h"

namespace DeepLearningCore
{
	class AffineLayerCore :
		public ILayerCore
	{
	public:
		AffineLayerCore(MatrixXX* pW, VectorXX* pB);
		virtual ~AffineLayerCore();
		virtual MatrixXX Forward(MatrixXX x);
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y);
		virtual LayerBackwardOutput Backward(MatrixXX dout);
		MatrixXX dw() { return _dw; }
		VectorXX db() { return _db; }
	private:
		int _numInput = 1;
		MatrixXX* _pW = NULL;
		VectorXX* _pB = NULL;
		MatrixXX _x;
		MatrixXX _dw;
		VectorXX _db;
	};
}
