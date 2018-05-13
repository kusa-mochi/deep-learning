#include "stdafx.h"
#include "CppUnitTest.h"
#include "TestUtility.h"

#include "IdentityLayerCore.h"
#include "AddLayerCore.h"
#include "MulLayerCore.h"
#include "AffineLayerCore.h"
#include "SigmoidLayerCore.h"
#include "ReLULayerCore.h"
#include "SoftmaxWithLoss.h"

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

		TEST_METHOD(SigmoidLayerForwardTest)
		{
			//SigmoidLayerCore layer;
			//MatrixXX a = Matrix<WEIGHT_TYPE, 5, 3>();
			//a <<
			//	1.0, 2.0, 3.0,
			//	4.0, 5.0, 6.0,
			//	7.0, 8.0, 9.0,
			//	10.0, 11.0, 12.0,
			//	13.0, 14.0, 15.0;
			//MatrixXX result = layer.Forward(a);

			//TestUtility util;
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(0, 0)), result(0, 0)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(0, 1)), result(0, 1)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(0, 2)), result(0, 2)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(1, 0)), result(1, 0)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(1, 1)), result(1, 1)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(1, 2)), result(1, 2)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(2, 0)), result(2, 0)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(2, 1)), result(2, 1)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(2, 2)), result(2, 2)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(3, 0)), result(3, 0)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(3, 1)), result(3, 1)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(3, 2)), result(3, 2)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(4, 0)), result(4, 0)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(4, 1)), result(4, 1)));
			//Assert::IsTrue(util.AreEqual(util.Sigmoid(a(4, 2)), result(4, 2)));
		}

		TEST_METHOD(GradientCheckTest)
		{
			Logger::WriteMessage("Begin GradientCheckTest");
			TestUtility util;

			LayerInfo* layerInfo = new LayerInfo[3];
			layerInfo[0].NumNeuron = 3;
			layerInfo[0].LayerType = _LayerType::Sigmoid;
			layerInfo[1].NumNeuron = 2;
			layerInfo[1].LayerType = _LayerType::Sigmoid;
			layerInfo[2].NumNeuron = 2;
			layerInfo[2].LayerType = _LayerType::SoftMax;

			MultiLayerPerceptronCore p(2, 3, layerInfo), q(2, 3, layerInfo);

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
			q.SetWeights(weights);

			// ÉoÉCÉAÉXÇÃê›íË
			WEIGHT_TYPE** bias = new WEIGHT_TYPE*[3]{
				new WEIGHT_TYPE[3]{ 0.1, 0.2, 0.3 },
				new WEIGHT_TYPE[2]{ 0.1, 0.2 },
				new WEIGHT_TYPE[2]{ 0.1, 0.2 }
			};
			p.SetBias(bias);
			q.SetBias(bias);

			// ì¸óÕÇÃê›íË
			WEIGHT_TYPE** x = new WEIGHT_TYPE*[2]{
				new WEIGHT_TYPE[2]{ 1.0, 0.5 },
				new WEIGHT_TYPE[2]{ 0.8, 0.7 }
			};

			// ã≥étêMçÜÇÃê›íË
			WEIGHT_TYPE** t = new WEIGHT_TYPE*[2]{
				new WEIGHT_TYPE[2]{ 0.9, 0.45 },
				new WEIGHT_TYPE[2]{ 0.75, 0.3 }
			};

			// êîílî˜ï™Ç≈åvéZÇµÇΩèdÇ›ÇÃå˘îzÇäiî[Ç∑ÇÈïœêî
			WEIGHT_TYPE*** numericdW = NULL;
			// êîílî˜ï™Ç≈åvéZÇµÇΩÉoÉCÉAÉXÇÃå˘îzÇäiî[Ç∑ÇÈïœêî
			WEIGHT_TYPE*** numericdB = NULL;
			// åÎç∑ãtì`îdñ@Ç≈åvéZÇµÇΩèdÇ›ÇÃå˘îzÇäiî[Ç∑ÇÈïœêî
			WEIGHT_TYPE*** dW = NULL;
			// åÎç∑ãtì`îdñ@Ç≈åvéZÇµÇΩÉoÉCÉAÉXÇÃå˘îzÇäiî[Ç∑ÇÈïœêî
			WEIGHT_TYPE*** dB = NULL;

			int numData = 2;
			p.DebugNumericGradient(x, t, numData, &numericdW, &numericdB);
			q.DebugGradient(x, t, numData, &dW, &dB);

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
						Assert::IsTrue(util.AreEqual(numericGradientWeight, gradientWeight), L"util.AreEqual(numericGradientWeight, gradientWeight)");
					}
					WEIGHT_TYPE numericGradientBias = numericdB[iLayer][0][iColumn];
					WEIGHT_TYPE gradientBias = dB[iLayer][0][iColumn];
					Assert::IsTrue(util.AreEqual(numericGradientBias, gradientBias), L"util.AreEqual(numericGradientBias, gradientBias)");
				}
			}

			Logger::WriteMessage("End GradientCheckTest");
		}
	};
}
