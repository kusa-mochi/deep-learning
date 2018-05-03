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

			LayerInfo* layerInfo = new LayerInfo[3];
			layerInfo[0].NumNeuron = 3;
			layerInfo[0].LayerType = _LayerType::Sigmoid;
			layerInfo[1].NumNeuron = 2;
			layerInfo[1].LayerType = _LayerType::Sigmoid;
			layerInfo[2].NumNeuron = 2;
			layerInfo[2].LayerType = _LayerType::None;

			MultiLayerPerceptronCore p(
				2,
				3,
				layerInfo
			);

			// 重みの設定
			WEIGHT_TYPE*** weights = new WEIGHT_TYPE**[3]{
				new WEIGHT_TYPE*[2] {
					new WEIGHT_TYPE[3]{ 0.1, 0.3, 0.5 },
					new WEIGHT_TYPE[3]{ 0.2, 0.4, 0.6 }
				},
				new WEIGHT_TYPE*[3]{
					new WEIGHT_TYPE[2]{ 0.1, 0.4 },
					new WEIGHT_TYPE[2]{ 0.2, 0.5 },
					new WEIGHT_TYPE[2]{ 0.3, 0.6 }
				},
				new WEIGHT_TYPE*[2]{
					new WEIGHT_TYPE[2]{ 0.1, 0.3 },
					new WEIGHT_TYPE[2]{ 0.2, 0.4 }
				}
			};
			p.SetWeights(weights);

			// バイアスの設定
			WEIGHT_TYPE** bias = new WEIGHT_TYPE*[3]{
				new WEIGHT_TYPE[3]{ 0.1, 0.2, 0.3 },
				new WEIGHT_TYPE[2]{ 0.1, 0.2 },
				new WEIGHT_TYPE[2]{ 0.1, 0.2 }
			};
			p.SetBias(bias);

			// 入力の設定
			WEIGHT_TYPE** coreInput = new WEIGHT_TYPE*[1]{
				new WEIGHT_TYPE[2]{ 1.0, 0.5 }
			};

			WEIGHT_TYPE** coreOutput = NULL;

			// 計算する
			p.Predict(
				coreInput,
				1,
				&coreOutput
			);

			Assert::IsTrue(0.31682707 < coreOutput[0][0] && coreOutput[0][0] < 0.31682709);
			Assert::IsTrue(0.69627908 < coreOutput[0][1] && coreOutput[0][1] < 0.69627910);
			Logger::WriteMessage("End SmallCaseCoreTest");
		}
	};
}
