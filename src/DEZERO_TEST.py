import unittest 
import numpy as np 

# 相対インポートなのでfrom-scratchディレクトリから以下のコマンドで実行する
# python3 -m DEZERO_TEST
from dezero import Variable
# from dezero.core_simple import Variable

x = Variable(np.array(1.0))
print('OK')

class Dezero_Test(unittest.TestCase):
    def test_package(self):
        x = Variable(np.array(1.0))
        self.assertEqual(x.data, np.array(1.0))

unittest.main()