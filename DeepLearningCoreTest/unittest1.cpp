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

			// èdÇ›ÇÃê›íË
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

			// ÉoÉCÉAÉXÇÃê›íË
			WEIGHT_TYPE** bias = new WEIGHT_TYPE*[3]{
				new WEIGHT_TYPE[3]{ 0.1, 0.2, 0.3 },
				new WEIGHT_TYPE[2]{ 0.1, 0.2 },
				new WEIGHT_TYPE[2]{ 0.1, 0.2 }
			};
			p.SetBias(bias);

			// ì¸óÕÇÃê›íË
			WEIGHT_TYPE** coreInput = new WEIGHT_TYPE*[1]{
				new WEIGHT_TYPE[2]{ 1.0, 0.5 }
			};

			WEIGHT_TYPE** coreOutput = NULL;

			// åvéZÇ∑ÇÈ
			p.Predict(
				coreInput,
				1,
				&coreOutput
			);

			Assert::IsTrue(0.31682707 < coreOutput[0][0] && coreOutput[0][0] < 0.31682709);
			Assert::IsTrue(0.69627908 < coreOutput[0][1] && coreOutput[0][1] < 0.69627910);
			Logger::WriteMessage("End SmallCaseCoreTest");
		}

		TEST_METHOD(GradientCheckTest)
		{
			Logger::WriteMessage("Begin GradientCheckTest");

			WEIGHT_TYPE allowableError = 0.5;

			LayerInfo* layerInfo = new LayerInfo[3];
			layerInfo[0].NumNeuron = 3;
			layerInfo[0].LayerType = _LayerType::Sigmoid;
			layerInfo[1].NumNeuron = 2;
			layerInfo[1].LayerType = _LayerType::Sigmoid;
			layerInfo[2].NumNeuron = 2;
			layerInfo[2].LayerType = _LayerType::SoftMax;

			MultiLayerPerceptronCore p(
				2,
				3,
				layerInfo
			);

			// èdÇ›ÇÃê›íË
			WEIGHT_TYPE*** weights = new WEIGHT_TYPE**[3]{
				new WEIGHT_TYPE*[2]{
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

			// ÉoÉCÉAÉXÇÃê›íË
			WEIGHT_TYPE** bias = new WEIGHT_TYPE*[3]{
				new WEIGHT_TYPE[3]{ 0.1, 0.2, 0.3 },
				new WEIGHT_TYPE[2]{ 0.1, 0.2 },
				new WEIGHT_TYPE[2]{ 0.1, 0.2 }
			};
			p.SetBias(bias);

			// ì¸óÕÇÃê›íË
			WEIGHT_TYPE** x = new WEIGHT_TYPE*[1]{
				new WEIGHT_TYPE[2]{ 1.0, 0.5 }
			};

			// ã≥étêMçÜÇÃê›íË
			WEIGHT_TYPE** t = new WEIGHT_TYPE*[1]{
				new WEIGHT_TYPE[2]{ 0.9, 0.45 }
			};

			// êîílî˜ï™Ç≈åvéZÇµÇΩèdÇ›ÇÃå˘îzÇäiî[Ç∑ÇÈïœêî
			WEIGHT_TYPE*** numericdW = NULL;
			// êîílî˜ï™Ç≈åvéZÇµÇΩÉoÉCÉAÉXÇÃå˘îzÇäiî[Ç∑ÇÈïœêî
			WEIGHT_TYPE*** numericdB = NULL;
			// åÎç∑ãtì`îdñ@Ç≈åvéZÇµÇΩèdÇ›ÇÃå˘îzÇäiî[Ç∑ÇÈïœêî
			WEIGHT_TYPE*** dW = NULL;
			// åÎç∑ãtì`îdñ@Ç≈åvéZÇµÇΩÉoÉCÉAÉXÇÃå˘îzÇäiî[Ç∑ÇÈïœêî
			WEIGHT_TYPE*** dB = NULL;

			int numData = 1;
			p.DebugNumericGradient(x, t, numData, &numericdW, &numericdB);
			p.DebugGradient(x, t, numData, &dW, &dB);

			Assert::AreEqual(p.GetNumLayer(), 3);
			Assert::AreEqual(p.GetNumInput(), 2);
			for (int iLayer = 0; iLayer < 3; iLayer++)
			{
				int rows = iLayer == 0 ? p.GetNumInput() : layerInfo[iLayer - 1].NumNeuron;
				int cols = layerInfo[iLayer].NumNeuron;

				for (int iColumn = 0; iColumn < cols; iColumn++)
				{
					for (int iRow = 0; iRow < rows; iRow++)
					{
						WEIGHT_TYPE numericGradientWeight = numericdW[iLayer][iRow][iColumn];
						WEIGHT_TYPE gradientWeight = dW[iLayer][iRow][iColumn];
						WEIGHT_TYPE diffdW = gradientWeight - numericGradientWeight;
						Assert::IsTrue(-allowableError < diffdW && diffdW < allowableError, L"-allowableError < diffdW && diffdW < allowableError");
					}
					WEIGHT_TYPE numericGradientBias = numericdB[iLayer][0][iColumn];
					WEIGHT_TYPE gradientBias = dB[iLayer][0][iColumn];
					WEIGHT_TYPE diffdB = gradientBias - numericGradientBias;
					Assert::IsTrue(-allowableError < diffdB && diffdB < allowableError, L"-allowableError < diffdB && diffdB < allowableError");
				}
			}

			Logger::WriteMessage("End GradientCheckTest");
		}
	};
}
