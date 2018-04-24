#include "stdafx.h"
#include "MultiLayerPerceptron.h"


namespace DeepLearning
{
	MultiLayerPerceptron::MultiLayerPerceptron(
		int numInput,											// “ü—Í‚ÌŸŒ³”
		cli::array<int>^ numNeuron,								// Še‘w‚Ìƒjƒ…[ƒƒ“‚Ì”
		ActivationFunctionType activationFunctionType,			// ’†ŠÔ‘w‚ÌŠˆ«‰»ŠÖ”
		OutputActivationFunctionType outputActivationFunctionType		// o—Í‘w‚ÌŠˆ«‰»ŠÖ”
	)
	{
		if (numInput < 1) throw gcnew System::ArgumentOutOfRangeException("numInput");
		if (numNeuron == nullptr) throw gcnew System::ArgumentNullException("numNeuron");

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
	}


	MultiLayerPerceptron::~MultiLayerPerceptron()
	{
	}

	void MultiLayerPerceptron::SetWeights(cli::array<WEIGHT_TYPE, 3>^ weights)
	{
		if (weights == nullptr) throw gcnew System::ArgumentNullException("weights");

		int length0 = weights->GetLength(0);
		int length1 = weights->GetLength(1);
		int length2 = weights->GetLength(2);
		WEIGHT_TYPE*** weightsPointer = NULL;

		weightsPointer = new WEIGHT_TYPE**[length0];
		for (int i = 0; i < length0; i++)
		{
			weightsPointer[i] = new WEIGHT_TYPE*[length1];
			for (int j = 0; j < length1; j++)
			{
				weightsPointer[i][j] = new WEIGHT_TYPE[length2];
				for (int k = 0; k < length2; k++)
				{
					weightsPointer[i][j][k] = weights[i, j, k];
				}
			}
		}

		_multiLayerPerceptronCore->SetWeights(weightsPointer);

		for (int i = 0; i < length0; i++)
		{
			for (int j = 0; j < length1; j++)
			{
				delete[] weightsPointer[i][j];
			}
			delete[] weightsPointer[i];
		}
		delete[] weightsPointer;
	}

	cli::array<WEIGHT_TYPE, 2>^ MultiLayerPerceptron::Predict(cli::array<WEIGHT_TYPE, 2>^ input)
	{
		if (input == nullptr) throw gcnew System::ArgumentNullException("input");

		WEIGHT_TYPE** coreInput = NULL;
		WEIGHT_TYPE** coreOutput = NULL;
		int inputRows = input->GetLength(0);

		this->ManagedArray2NativeArray(input, coreInput);

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

	void MultiLayerPerceptron::ManagedArray2NativeArray(cli::array<WEIGHT_TYPE, 2>^ input, WEIGHT_TYPE** output)
	{
		int inputRows = input->GetLength(0);
		int inputCols = input->GetLength(1);

		output = new WEIGHT_TYPE*[inputRows];
		for (int i = 0; i < inputRows; i++)
		{
			output[i] = new WEIGHT_TYPE[inputCols];
			for (int j = 0; j < inputCols; j++)
			{
				output[i][j] = input[i, j];
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
