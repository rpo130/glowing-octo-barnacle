# author: 易森林
# email: ysl.130@qq.com

from fractions import Fraction
from math import sqrt
import random
from typing import List
import unittest
import copy

#范数类型，可选值['2', 'inf']
NORMNUM = 'inf' 
#误差值
EPS = 0.00001
PRICISION=4

#向量类
#元素类型为float
class YVec:
    def __init__(self, element):
        assert len(element) != 0, "not match"
        self.element = []
        for e in element:
            self.element.append(round(Fraction(e), PRICISION))

    def __len__(self):
        return len(self.element)

    def __str__(self) -> str:
        return str(self.element)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        
        if not isinstance(other, YVec):
            return False

        if len(self.element) != len(other.element):
            return False

        for i in range(len(self.element)):
            if self.element[i] != other.element[i]:
                return False

        return True

    def print_str(self):
        return str(list(map(lambda x:float(x), self.element)))

    #标量 乘 向量
    @staticmethod
    def scalarMul(n : int, v1):
        assert len(v1) != 0, "not match"
        r = []
        for i in v1.element:
            r.append(n * i)
        return YVec(r)

    #向量 加 向量
    @staticmethod
    def add(v1, v2):
        assert len(v1) == len(v2), "not match"
        r = []
        for i in range(len(v1)):
            r.append(v1.element[i] + v2.element[i])
        return YVec(r)

    #向量 减 向量
    @staticmethod
    def minus(v1, v2):
        assert len(v1) == len(v2), "not match"
        r = []
        for i in range(len(v1)):
            r.append(v1.element[i] - v2.element[i])
        return YVec(r)

    #向量相似比较，越低越接近
    @staticmethod
    def elementLoss(v1, v2):
        assert len(v1) == len(v2), "not match"
        loss = 0.0
        for i in range(len(v1)):
            loss += abs(v1.element[i] - v2.element[i])
        return loss

#矩阵
class YMatrix:
    #不允许为空矩阵
    def __init__(self, vecs):
        n = len(vecs)
        self.vecs = []
        l = len(vecs[0])
        for v in vecs:
            if l != len(v):
                raise Exception('num not match')
            self.vecs.append(YVec(v.element))

    def print(self):
        for v in self.vecs:
            print("vector " + str(v))

    def print_str(self):
        r = []
        for v in self.vecs:
            r.append(v.print_str())
        return r

    #行为向量个数，列为向量维数
    def shape(self):
        r = len(self.vecs)
        c = len(self.vecs[0])
        return (r,c)

#范数计算
#v1：向量
#n：范数选项
def yNorm(v1 : YVec, n : str) -> Fraction:
    assert v1 is not None, "not match"
    assert len(v1) != 0, "not match"

    if n == 'inf':
        m = 0
        for v in v1.element:
            if abs(v) > m:
                m = abs(v)
        return Fraction(round(m,PRICISION))

    if n == '2':
        s = 0
        for v in v1.element:
            s += v**2
        return Fraction(round(sqrt(s),PRICISION))

    raise Exception('not match')

#矩阵 乘 向量
def yMatrixMulVec(m1 : YMatrix, v1 : YVec) -> YVec:
    assert len(m1.vecs) == len(v1), "not match"

    r = None
    for i in range(len(m1.vecs)):
        if r is None :
            r = YVec.scalarMul(v1.element[i], m1.vecs[i])
        else:
            r = YVec.add(r, YVec.scalarMul(v1.element[i], m1.vecs[i]))

    return r

#向量内积
def yInnerProduct(v1 : YVec, v2 : YVec) -> Fraction:
    assert len(v1) == len(v2), "not match"

    r = 0
    for i in range(len(v1)):
        r += (v1.element[i] * v2.element[i])

    return Fraction(round(r,5))

#矩阵主特征计算
def findDominat(m1 : YMatrix):
    r, c = m1.shape()
    v0 = yRandomGenUnitVec(c)
    return findDominatWithInit(m1, v0)

#矩阵主特征计算
def findDominatWithInit(m1 : YMatrix, vinit : YVec):
    T = 1000
    
    #init v
    v0 = vinit

    u0 = YVec.scalarMul(1/yNorm(v0, NORMNUM), v0)
    
    #A vo
    v = yMatrixMulVec(m1, u0)
    u = YVec.scalarMul(1/yNorm(v, NORMNUM), v)
    i = 0
    upre = u0
    while yNorm(YVec.minus(u,upre), NORMNUM) > EPS and i < T:
        upre = u
        v = yMatrixMulVec(m1, upre)
        u = YVec.scalarMul(1/yNorm(v, NORMNUM), v)

        i += 1

    # return (yNorm(v, NORMNUM), u)
    return (yInnerProduct(yMatrixMulVec(m1, u), u) / yInnerProduct(u,u), u)

#随机产生单一向量
#n为维度
def yRandomGenVec(n : int):
    r = []
    for i in range(n):
        r.append(random.randrange(-10, 10))

    return YVec(r)

def yUnitVec(v : YVec):
    assert len(v) > 0
    
    s = 0
    for e in v.element:
        s += e**2
    s = sqrt(s)
    r = []
    for e in v.element:
        r.append(e/s)
    return YVec(r)

def yRandomGenUnitVec(n : int):
    v = yRandomGenVec(n)
    return yUnitVec(v)

#正交向量计算
#vecs: 需要被正交的向量
#return: 返回的向量和vecs里的所有向量正交
def yOrthogonalVec(vecs : List[YVec]) :
    nvecs = copy.deepcopy(vecs)

    n = len(nvecs[0])
    randVec = yRandomGenVec(n)

    nvecs.append(randVec)

    orthVecs = []

    for e in nvecs:
        v = copy.copy(e)
        v = YVec(list(map(lambda x:round(x, PRICISION), v.element)))
        for o in orthVecs:
            v = YVec.minus(v, YVec.scalarMul(yInnerProduct(e, o) / yInnerProduct(o, o), o)) 
        orthVecs.append(v)

    return orthVecs[len(nvecs)-1]

#标准化，单位向量
def yOrthogonalUnitVec(vecs : List[YVec]) :
    v = yOrthogonalVec(vecs)
    v = yUnitVec(v)
    #first non zero num be positive
    need_flip = False
    
    r = v.element

    for i in r:
        if i == 0:
            continue
        if i < 0:
            need_flip = True
        else:
            break
    
    if need_flip:
        r = list(map(lambda x : x * -1, r))
    return YVec(r)

#特征值，特征向量计算
def eigen(m1 : YMatrix):
    r = []

    _, c = m1.shape()

    showVec = []
    v0 = None
    for i in range(c):
        if len(showVec) == 0:
            v0 = yRandomGenUnitVec(c)
        else:
            v0 = yOrthogonalUnitVec(showVec)
            v0 = YVec(list(map(lambda x : round(x, PRICISION), v0.element)))

            # print(f'showVec: {showVec}')
            # print(f'v0: {v0}')
        evalue, evector = findDominatWithInit(m1, v0)
        
        #精度问题处理
        evalue = round(evalue, PRICISION)
        evector = YVec(list(map(lambda x : round(x, PRICISION), evector.element)))
        
        showVec.append(evector)
        r.append((float(evalue), evector.print_str()))

    return r

if __name__=="__main__":
    input = YMatrix([YVec([5, 12]), YVec([12,12])])
    print(f"input : {input.print_str()}")
    print(eigen(input))

    input = YMatrix([YVec([1, 0, 0]), YVec([0, 2, 0]), YVec([0, 0, 4])])
    print(f"input : {input.print_str()}")
    print(eigen(input))

    input = YMatrix([YVec([1, 0, 0,0]), YVec([0, 2, 0,0]), YVec([0, 0, 4,0]), YVec([0,0,0,9])])
    print(f"input : {input.print_str()}")
    print(eigen(input ))
