#pragma once

#define ARGUMENT_EXCEPTION 1
#define ARGUMENT_NULL_EXCEPTION 2
#define INVALID_OPERATION_EXCEPTION 3
#define NOT_IMPLEMENTED_EXCEPTION 4;
typedef double WEIGHT_TYPE;

namespace DeepLearningCore
{
	enum _LayerType
	{
		None = 0,
		Add,
		Mul,
		Affine,
		Sigmoid,
		ReLU,
		SoftMax
	};

	typedef struct ST_LayerInfo
	{
		int NumNeuron;
		_LayerType LayerType;	// äàê´âªä÷êîÇÃéÌóﬁ
	} LayerInfo;
}
