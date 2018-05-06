using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UnitTestDeepLearning
{
    public class TestUtilities
    {
        public bool Double2Bool(double d)
        {
            return d > 0.5;
        }

        public bool IsEqualDouble(double d1, double d2)
        {
            return (
                (this.Double2Bool(d1) && this.Double2Bool(d2)) ||
                (!this.Double2Bool(d1) && !this.Double2Bool(d2))
                );
        }
    }
}
