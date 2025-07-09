import unittest
from multivar_threepart import *

class TestMultiVar(unittest.TestCase):
    def test_parser(self):
        filename = "5mA1msec-0.2mA24msec0-12hour"
        print(extract_parameters(filename))
        self.assertEqual(extract_parameters(filename),(5.0,1,0.2,24,0,12))

if __name__ == '__main__':
    unittest.main()