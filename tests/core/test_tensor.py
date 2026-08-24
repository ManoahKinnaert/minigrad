"""
This file houses unittests for the minigrad.core.Tensor class.
"""
import unittest
from minigrad import Tensor

class TensorTest(unittest.TestCase):
    def test_init_value_error(self):
        """Test data Nonetype""" 
        self.assertRaises(ValueError, lambda: Tensor(None, None))

if __name__ == "__main__":
    unittest.main()