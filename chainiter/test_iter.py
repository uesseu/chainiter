import unittest
from chainiter import ChainIter, future, ProgressBar, curry, logger, start_async, run_async, run_coroutine
from functools import partial
from doctest import testmod
from time import sleep, time
from functools import partial
import asyncio
from logging import getLogger, basicConfig, INFO, WARNING, StreamHandler
from itertools import product
basicConfig(level=INFO)
# logger.addHandler(StreamHandler())

def normal_func(x: int) -> int: return x * 2
def normal_star_func(x: int, y: int) -> int: return x * y
@future
async def hoge(x: int) -> int: return x * 3
async def async_func(x: int) -> int: return await hoge(x)
async def fuga(x: int, y: int) -> int: return x * y
async def async_star_func(x: int, y: int) -> int: return await fuga(x, y)

class TestStringMethods(unittest.TestCase):
    def test_curry(self):
        @curry(2)
        def mm(x, y): return x * y
        self.assertEqual(mm(9)(8), 72)
    def test_normal_map(self):
        self.assertEqual(ChainIter([1, 2, 3]).map(
            normal_func).get(), [2, 4, 6])
    def test_async_map(self):
        self.assertEqual(ChainIter([1, 2, 3]).map(async_func).get(), [3, 6, 9])
    def test_star_map(self):
        self.assertEqual(ChainIter([[1, 2], [3, 4]]).starmap(
            normal_star_func).get(), [2, 12])
    def test_async_star_map(self):
        self.assertEqual(ChainIter([[1, 2], [3, 4]]).starmap(
            async_star_func).get(), [2, 12])
    def test_normal_map_multi(self):
        self.assertEqual(ChainIter([1, 2, 3]).map(
            normal_func, 2).get(), [2, 4, 6])
    def test_async_map_multi(self):
        self.assertEqual(ChainIter([1, 2, 3]).map(
            async_func, 2).get(), [3, 6, 9])
    def test_star_map_multi(self):
        self.assertEqual(ChainIter([[1, 2], [3, 4]]).starmap(
            normal_star_func, 2).get(), [2, 12])
    def test_async_star_map_multi(self):
        self.assertEqual(ChainIter([[1, 2], [3, 4]]).starmap(
            async_star_func, 2).get(), [2, 12])

def speed_test():
    print('first')
    current = time()
    ChainIter(range(10000)).map(normal_func).calc()
    print(time() - current)

    print('second')
    current = time()
    [normal_func(n) for n in range(10000)]
    print(time() - current)

####################
# Async speed test
####################
def async_speed_test():
    current = time()
    ChainIter(range(10000)).map(normal_func).calc()
    print(time() - current)

    current = time()
    ChainIter(range(10000)).map(normal_func, 5).calc()
    print(time() - current)
####################

speed_test()
async_speed_test()
if __name__ == '__main__':
    unittest.main()
