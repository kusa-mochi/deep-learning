#pragma once

namespace DeepLearning
{
	//None = 0,
	//	Add,
	//	Mul,
	//	Affine,
	//	Sigmoid,
	//	ReLU,
	//  SoftMax
	public enum class LayerType : int { None, Add, Mul, Affine, Sigmoid, ReLU, SoftMax };
}
