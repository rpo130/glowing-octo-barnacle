from cProfile import run
from fractions import Fraction
import unittest
from math import sqrt
from yeigen import YMatrix, YVec, yInnerProduct, yMatrixMulVec, yNorm, yOrthogonalUnitVec, yOrthogonalVec, yRandomGenUnitVec, PRICISION

class TestEigen(unittest.TestCase):

    def test_yvec(self):
        yvec = YVec([4,5,6])
        self.assertEqual(3, len(yvec))
        self.assertEqual(str([Fraction(4),Fraction(5),Fraction(6)]), str(yvec))

        self.assertEqual(YVec([4,8,12]), YVec.scalarMul(4, YVec([1,2,3])))
        self.assertEqual(YVec([-8,-3,-1,-6]), YVec.scalarMul(-1, YVec([8,3,1,6])))

        self.assertEqual(YVec([5,7,9]), YVec.add(YVec([1,2,3]), YVec([4,5,6])))
        self.assertEqual(YVec([-3,-3,-3]), YVec.minus(YVec([1,2,3]), YVec([4,5,6])))


    def test_ymatrix(self):
        self.assertEqual((3,3), YMatrix([YVec([1,2,3]), YVec([1,1,1]), YVec([2,3,4])]).shape())
        self.assertEqual((3,2), YMatrix([YVec([1,2]), YVec([1,1]), YVec([2,3])]).shape())

    def test_ynorm(self):
        self.assertEqual(5, yNorm(YVec([3,4]), '2'))
        self.assertEqual(Fraction(round(sqrt(4**2+5**2+6**2), PRICISION)), yNorm(YVec([4,5,6]), '2'))

        self.assertEqual(2, yNorm(YVec([1,-2]), 'inf'))
        self.assertEqual(7, yNorm(YVec([4,5,7]), 'inf'))

    def test_yMatrixMulVec(self):
        self.assertEqual(YVec([7, 10, 13]), yMatrixMulVec(YMatrix([YVec([1,2,3]), YVec([1,1,1]), YVec([2,3,4])]), YVec([2,3,1])))

    def test_yInnerProduct(self):
        self.assertEqual(20, yInnerProduct(YVec([2,3,4]), YVec([5,2,1])))

    def test_yRandomGenUnitVec(self):
        v = yRandomGenUnitVec(4)
        s = 0
        for i in v.element:
            s += i**2
        self.assertTrue((1-s)<0.00001)

    def test_findDominat(self):
        pass

    def test_yOrthogonalVec(self):
        self.assertEqual(YVec([0,0,1]), yOrthogonalUnitVec([YVec([1,0,0]), YVec([0,1,0])]))

    def test_eigen(self):
        pass

if __name__ == "__main__":
    unittest.main()