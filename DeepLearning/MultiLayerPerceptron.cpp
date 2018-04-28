#include "stdafx.h"
#include "MultiLayerPerceptron.h"

namespace DeepLearning
{
	MultiLayerPerceptron::MultiLayerPerceptron(
		int numInput,											// 入力の次元数
		cli::array<int>^ numNeuron,								// 各層のニューロンの数
		ActivationFunctionType activationFunctionType,			// 中間層の活性化関数
		OutputActivationFunctionType outputActivationFunctionType		// 出力層の活性化関数
	)
	{
		if (numInput < 1) throw gcnew System::ArgumentOutOfRangeException("numInput");
		if (numNeuron == nullptr) throw gcnew System::ArgumentNullException("numNeuron");

		System::Diagnostics::Trace::TraceInformation("new MultiLayerPerceptron: ガード節通過");

		_numNeuron = numNeuron;

		int* numNeuronPointer = new int[numNeuron->Length];
		for (int i = 0; i < numNeuron->Length; i++) numNeuronPointer[i] = numNeuron[i];

		int activationFunctionInt = -1;
		int outputActivationFunctionInt = -1;

		switch (activationFunctionType)
		{
		case ActivationFunctionType::Sigmoid:
			activationFunctionInt = FUNCTION_SIGMOID;
			break;
		case ActivationFunctionType::ReLU:
			activationFunctionInt = FUNCTION_RELU;
			break;
		}

		switch (outputActivationFunctionType)
		{
		case OutputActivationFunctionType::None:
			outputActivationFunctionInt = FUNCTION_NONE;
			break;
		case OutputActivationFunctionType::SoftMax:
			outputActivationFunctionInt = FUNCTION_SOFTMAX;
			break;
		case OutputActivationFunctionType::Sigmoid:
			outputActivationFunctionInt = FUNCTION_SIGMOID;
			break;
		case OutputActivationFunctionType::ReLU:
			outputActivationFunctionInt = FUNCTION_RELU;
			break;
		}

		_multiLayerPerceptronCore = new MultiLayerPerceptronCore(
			numInput,
			numNeuron->Length,
			numNeuronPointer,
			activationFunctionInt,
			outputActivationFunctionInt
		);

		delete[] numNeuronPointer;
	}


	MultiLayerPerceptron::~MultiLayerPerceptron()
	{
		delete _multiLayerPerceptronCore;
	}

	void MultiLayerPerceptron::SetWeights(cli::array<cli::array<WEIGHT_TYPE, 2>^>^ weights)
	{
		if (weights == nullptr) throw gcnew System::ArgumentNullException("weights");

		int Layers = weights->Length;
		WEIGHT_TYPE*** weightsPointer = new WEIGHT_TYPE**[Layers];
		for (int i = 0; i < Layers; i++)
		{
			int Rows = weights[i]->GetLength(0);
			int Columns = weights[i]->GetLength(1);
			weightsPointer[i] = new WEIGHT_TYPE*[Rows];
			for (int j = 0; j < Rows; j++)
			{
				weightsPointer[i][j] = new WEIGHT_TYPE[Columns];
				for (int k = 0; k < Columns; k++)
				{
					weightsPointer[i][j][k] = weights[i][j, k];
				}
			}
		}

		_multiLayerPerceptronCore->SetWeights(weightsPointer);

		Layers = weights->Length;
		for (int i = 0; i < Layers; i++)
		{
			int Rows = weights[i]->GetLength(0);
			for (int j = 0; j < Rows; j++)
			{
				delete[] weightsPointer[i][j];
			}
			delete[] weightsPointer[i];
		}
		delete[] weightsPointer;
	}

	void MultiLayerPerceptron::SetBias(cli::array<cli::array<WEIGHT_TYPE>^>^ bias)
	{
		if (bias == nullptr) throw gcnew System::ArgumentNullException("bias");

		int Layers = bias->Length;
		WEIGHT_TYPE** biasPointer = new WEIGHT_TYPE*[Layers];
		for (int i = 0; i < Layers; i++)
		{
			int Neurons = bias[i]->Length;
			biasPointer[i] = new WEIGHT_TYPE[Neurons];
			for (int j = 0; j < Neurons; j++)
			{
				biasPointer[i][j] = bias[i][j];
			}
		}

		_multiLayerPerceptronCore->SetBias(biasPointer);

		Layers = bias->Length;
		for (int i = 0; i < Layers; i++)
		{
			delete[] biasPointer[i];
		}
		delete[] biasPointer;
	}

	cli::array<WEIGHT_TYPE, 2>^ MultiLayerPerceptron::Predict(cli::array<WEIGHT_TYPE, 2>^ input)
	{
		if (input == nullptr) throw gcnew System::ArgumentNullException("input");

		WEIGHT_TYPE** coreInput = NULL;
		WEIGHT_TYPE** coreOutput = NULL;
		int inputRows = input->GetLength(0);

		this->ManagedArray2NativeArray(input, &coreInput);
		System::Diagnostics::Debug::Assert(coreInput != NULL);

		_multiLayerPerceptronCore->Predict(coreInput, inputRows, coreOutput);

		for (int i = 0; i < input->GetLength(0); i++)
		{
			delete[] coreInput[i];
		}
		delete[] coreInput;

		cli::array<WEIGHT_TYPE, 2>^ output = this->NativeArray2ManagedArray(
			coreOutput,
			inputRows,
			_multiLayerPerceptronCore->GetNumOutput()
		);

		for (int i = 0; i < output->GetLength(0); i++)
		{
			delete[] coreOutput[i];
		}
		delete[] coreOutput;

		return output;
	}

	void MultiLayerPerceptron::ManagedArray2NativeArray(cli::array<WEIGHT_TYPE, 2>^ input, WEIGHT_TYPE*** output)
	{
		int inputRows = input->GetLength(0);
		int inputCols = input->GetLength(1);

		*output = new WEIGHT_TYPE*[inputRows];
		for (int i = 0; i < inputRows; i++)
		{
			*output[i] = new WEIGHT_TYPE[inputCols];
			for (int j = 0; j < inputCols; j++)
			{
				*output[i][j] = input[i, j];
			}
		}
	}

	cli::array<WEIGHT_TYPE, 2>^ MultiLayerPerceptron::NativeArray2ManagedArray(WEIGHT_TYPE** input, int rows, int cols)
	{
		cli::array<WEIGHT_TYPE, 2>^ output = gcnew cli::array<WEIGHT_TYPE, 2>(
			rows,
			_multiLayerPerceptronCore->GetNumOutput()
			);

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				output[i, j] = input[i][j];
			}
		}

		return output;
	}
}
