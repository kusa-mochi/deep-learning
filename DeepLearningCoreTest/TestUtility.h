#pragma once

namespace DeepLearningCoreTest
{
	class TestUtility
	{
	public:
		TestUtility() {}
		virtual ~TestUtility() {}
		bool AreEqual(double d1, double d2)
		{
			return (
				(d1 * (1.0 - _doubleAccuracyLevel)) < d2 &&
				d2 < (d1 * _doubleAccuracyLevel)
				);
		}
		double Sigmoid(double d)
		{
			return (1.0 / (1.0 + std::exp(-d)));
		}
	private:
		double _doubleAccuracyLevel = 0.01;
	};
}
