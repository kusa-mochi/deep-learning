#pragma once
#include "stdafx.h"

public class MultilayerPerceptron
{
private:
	int _numLayer = 0;
	int* _numNeuron = NULL;
	double*** _weight = NULL;
	double(*_ActivationFunction)(double* d) = NULL;
	double(*_OutputActivationFunction)(double* d) = NULL;
public:
	MultilayerPerceptron(
		int numLayer,									// 入力層を除く，層の数
		int* numNeuron,									// 各層のニューロンの数
		double*** weight,								// i番目の層の，j番目のニューロンの，k番目のシナプスの重み
		double(*ActivationFunction)(double* d),			// 入力層・中間層の活性化関数,
		double(*OutputActivationFunction)(double* d)	// 出力層の活性化関数
	);
	virtual ~MultilayerPerceptron();
};
