#include "stdafx.h"
#include "CppUnitTest.h"
#include "MultiLayerPerceptronCore.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearningCore;

namespace DeepLearningCoreTest
{
	TEST_CLASS(MultiLayerPerceptronCoreTest)
	{
	public:

		TEST_METHOD(SmallCaseCoreTest)
		{
			Logger::WriteMessage("Begin SmallCaseCoreTest");

			int* numNeuron = new int[3]{ 3,2,2 };
			MultiLayerPerceptronCore* p = new MultiLayerPerceptronCore(
				2,
				3,
				numNeuron
			);

			delete[] numNeuron;
			numNeuron = NULL;
			delete p;
			p = NULL;

			Assert::AreEqual(1, 1);
			Logger::WriteMessage("End SmallCaseCoreTest");
		}

	};
}
