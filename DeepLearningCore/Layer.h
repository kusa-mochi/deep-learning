#pragma once
#include "ImportEigen.h"

namespace DeepLearningCore
{
	typedef struct ST_LayerBackwardOutput
	{
		MatrixXX x;
		MatrixXX y;
	} LayerBackwardOutput;

	class ILayerCore
	{
	public:
		virtual ~ILayerCore() {}
		virtual MatrixXX Forward(MatrixXX x) = 0;
		virtual MatrixXX Forward(MatrixXX x, MatrixXX y) = 0;
		virtual LayerBackwardOutput Backward(MatrixXX dout) = 0;
	};

	typedef struct ST_Layer
	{
		ILayerCore* Layer;
		_LayerType LayerType;
		struct ST_Layer* Next;
	} Layer;
}
