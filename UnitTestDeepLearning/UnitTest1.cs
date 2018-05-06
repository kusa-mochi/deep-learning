using System;
using System.Diagnostics;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using DeepLearning;

namespace UnitTestDeepLearning
{
    [TestClass]
    public class UnitTestMultiLayerPerceptron
    {
        [TestMethod]
        public void SmallCaseTest()
        {
            MultiLayerPerceptron p = new MultiLayerPerceptron(
                2,
                new int[] { 3, 2, 2 },
                new LayerType[] { LayerType.Sigmoid, LayerType.Sigmoid, LayerType.None }
                );
            p.SetWeights(
                new double[3][,]
                {
                    new double[,]{
                        { 0.1, 0.3, 0.5 },
                        { 0.2, 0.4, 0.6 }
                    },
                    new double[,]{
                        { 0.1, 0.4 },
                        { 0.2, 0.5 },
                        { 0.3, 0.6 }
                    },
                    new double[,]
                    {
                        { 0.1, 0.3 },
                        { 0.2, 0.4 }
                    }
                }
                );
            p.SetBias(
                new double[3][]
                {
                    new double[]{ 0.1, 0.2, 0.3 },
                    new double[]{ 0.1, 0.2 },
                    new double[]{ 0.1, 0.2 }
                }
                );
            var result = p.Predict(new double[,] { { 1.0, 0.5 } });

            // valid result: 0.31682708, 0.69627909
            Console.WriteLine("result: {0}, {1}", result[0, 0], result[0, 1]);
            Assert.IsTrue(0.31682707 < result[0, 0] && result[0, 0] < 0.31682709);
            Assert.IsTrue(0.69627908 < result[0, 1] && result[0, 1] < 0.69627910);
        }

        [TestMethod]
        public void TwoDimensionalPositionTest()
        {
            int numInput = 2;
            int numOutput = 2;

            MultiLayerPerceptron p = new MultiLayerPerceptron(
                numInput,
                new int[] { 3, 2, numOutput },
                new LayerType[] { LayerType.Sigmoid, LayerType.Sigmoid, LayerType.SoftMax }
                );

            // 学習データのバッチサイズ
            int numLearningData = 100;

            // ミニバッチ学習の回数
            int numBatch = 10000;

            // テストデータの個数
            int numTestData = 100;

            // 精度
            double[] accuracy = new double[numBatch];

            // 損失
            double[] loss = new double[numBatch];

            Random rnd = new Random(DateTime.Now.Millisecond);

            for (int iBatch = 0; iBatch < numBatch; iBatch++)
            {
                // 学習データ
                double[,] learningData = new double[numLearningData, numInput];
                // 教師データ
                double[,] teachData = new double[numLearningData, numOutput];
                for (int iData = 0; iData < numLearningData; iData++)
                {
                    // 0～1の間
                    learningData[iData, 0] = rnd.NextDouble();
                    learningData[iData, 1] = rnd.NextDouble();

                    if ((learningData[iData, 0] * learningData[iData, 0]) + (learningData[iData, 1] * learningData[iData, 1]) < 0.25)
                    {
                        teachData[iData, 0] = 0.0;
                        teachData[iData, 1] = 1.0;
                    }
                    else
                    {
                        teachData[iData, 0] = 1.0;
                        teachData[iData, 1] = 0.0;
                    }
                }

                // テストデータ
                double[,] testData = new double[numTestData, numInput];
                // テストデータに対応する答え
                double[,] validOutputData = new double[numTestData, numOutput];
                for (int iData = 0; iData < numTestData; iData++)
                {
                    // 0～1の間
                    testData[iData, 0] = rnd.NextDouble();
                    testData[iData, 1] = rnd.NextDouble();

                    if ((testData[iData, 0] * testData[iData, 0]) + (testData[iData, 1] * testData[iData, 1]) < 0.25)
                    {
                        validOutputData[iData, 0] = 0.0;
                        validOutputData[iData, 1] = 1.0;
                    }
                    else
                    {
                        validOutputData[iData, 0] = 1.0;
                        validOutputData[iData, 1] = 0.0;
                    }
                }

                // 学習する。
                p.Learn(learningData, teachData, 0.1);

                // 性能の確認のため，テストデータを入力してみる。
                double[,] result = p.Predict(testData);

                //int[] bunrui = new int[numTestData];

                // テストデータ入力に対する結果resultを，
                // 正しい答えであるvalidOutputDataに照らして評価する。
                int numCorrect = 0; // 正解数
                TestUtilities util = new TestUtilities();
                for (int iData = 0; iData < numTestData; iData++)
                {
                    if (
                        util.IsEqualDouble(result[iData, 0], validOutputData[iData, 0]) &&
                        util.IsEqualDouble(result[iData, 1], validOutputData[iData, 1])
                        )
                    {
                        // 正解
                        numCorrect++;
                    }

                    //if (util.Double2Bool(result[iData, 0]) && !util.Double2Bool(result[iData, 1]))
                    //{
                    //    bunrui[iData] = 1;
                    //}
                    //else if (!util.Double2Bool(result[iData, 0]) && util.Double2Bool(result[iData, 1]))
                    //{
                    //    bunrui[iData] = 0;
                    //}
                    //else
                    //{
                    //    bunrui[iData] = 2;
                    //}
                }
                accuracy[iBatch] = 100.0 * numCorrect / numTestData;

                Console.WriteLine("batch: {0}/{1}", iBatch + 1, numBatch);

                if (iBatch == numBatch - 1)
                {
                    Console.WriteLine("class1,class2,正解1,正解2");
                    for (int iData = 0; iData < numTestData; iData++)
                    {
                        Console.WriteLine("{0},{1},{2},{3}", result[iData, 0], result[iData, 1], validOutputData[iData, 0], validOutputData[iData, 1]);
                    }
                }
            }

            //// 精度の変化を標準出力する。
            //for (int iBatch = 0; iBatch < numBatch; iBatch++)
            //{
            //    Console.WriteLine("{0} {1}", iBatch, accuracy[iBatch]);
            //}
        }
    }
}
