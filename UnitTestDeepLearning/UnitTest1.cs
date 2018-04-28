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
                ActivationFunctionType.Sigmoid,
                OutputActivationFunctionType.None
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
            Assert.IsTrue(0.31682707 < result[0, 0] && result[0, 0] < 0.31682709);
            Console.WriteLine("result: {0}, {1}", result[0, 0], result[0, 1]);
        }
    }
}
