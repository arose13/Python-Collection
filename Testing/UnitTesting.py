# API Unit Testing
from Web.API import square, welcome_me, index, do_math
import unittest


class APIUnitTest(unittest.TestCase):
    def test_simple_functions(self):
        assert square(2) == 4
        assert index() == 'Oh Hello'
        assert welcome_me('Anthony') == 'Welcome Anthony'
        assert do_math(3) == '3 * 3 = 9'

if __name__ == '__main__':
    unittest.main()
