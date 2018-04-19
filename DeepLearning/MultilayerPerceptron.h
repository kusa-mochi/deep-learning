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
		int numLayer,									// ���͑w�������C�w�̐�
		int* numNeuron,									// �e�w�̃j���[�����̐�
		double*** weight,								// i�Ԗڂ̑w�́Cj�Ԗڂ̃j���[�����́Ck�Ԗڂ̃V�i�v�X�̏d��
		double(*ActivationFunction)(double* d),			// ���͑w�E���ԑw�̊������֐�,
		double(*OutputActivationFunction)(double* d)	// �o�͑w�̊������֐�
	);
	virtual ~MultilayerPerceptron();
};
