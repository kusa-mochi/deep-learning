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
                new int[] { 5, 3, 2 },
                new LayerType[] { LayerType.Sigmoid, LayerType.Sigmoid, LayerType.SoftMax }
                );

            // 学習データの個数
            int numLearningData = 1000;

            // テストデータの個数
            int numTestData = 1000;

            // 学習データを用意する。
            double[,] learningData = new double[numLearningData, numInput];

            // 教師データを用意する。
            double[,] teachData = new double[numLearningData, numOutput];

            // テストデータを用意する。
            double[,] testData = new double[numTestData, numInput];

            // テストデータに対応する答えを用意する。
            double[,] validOutputData = new double[numTestData, numOutput];

            // 学習する。
            p.Learn(learningData, teachData, 0.3);

            // 性能の確認のため，テストデータを入力してみる。
            double[,] result = p.Predict(testData);

            // テストデータ入力に対する結果resultを，
            // 正しい答えであるvalidOutputDataに照らして評価する。
        }
    }
}
