using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using DeepLearning;

namespace UnitTestDeepLearning
{
    [TestClass]
    public class UnitTestMultiLayerPerceptron
    {
        [TestMethod]
        public void TestMethod1()
        {
            MultiLayerPerceptron p = new MultiLayerPerceptron(
                2,
                new int[] { 2, 1 },
                ActivationFunctionType.Sigmoid,
                OutputActivationFunctionType.None
                );
        }
    }
}
