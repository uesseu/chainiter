import unittest
from chainiter import (ChainIter, future, ProgressBar, curry,
                       logger, start_async, run_async, run_coroutine,
                       thread_starmap, gen_map, gen_starmap)
from chainiter.pipeline import ThreadPipeMap
from functools import partial
from doctest import testmod
from time import sleep, time
from functools import partial
import asyncio
from logging import getLogger, basicConfig, INFO, WARNING, StreamHandler
from itertools import product
# basicConfig(level=INFO)
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
            async_func, thread=2).get(), [3, 6, 9])
        self.assertEqual(ChainIter([1, 2, 3]).map(
            async_func, 2).get(), [3, 6, 9])
    def test_star_map_multi(self):
        self.assertEqual(ChainIter([[1, 2], [3, 4]]).starmap(
            normal_star_func, thread=2).get(), [2, 12])
        self.assertEqual(ChainIter([[1, 2], [3, 4]]).starmap(
            normal_star_func, 2).get(), [2, 12])
    def test_async_star_map_multi(self):
        self.assertEqual(ChainIter([[1, 2], [3, 4]]).starmap(
            async_star_func, 2).get(), [2, 12])
        self.assertEqual(ChainIter([[1, 2], [3, 4]]).starmap(
            async_star_func, thread=2).get(), [2, 12])

    def test_pipe_starmap(self):
        tester(self, ThreadPipeMap(3).starmap(test, [(i,) for i in range(9)]).get())

    def test_ThreadPipeMap(self):
        tester(self, ThreadPipeMap(3).map(test, [i for i in range(9)]).get())

    def test_mp(self):
        res = gen_map(test, [i for i in range(9)], 3)
        tester(self, res)
        res = gen_map(test, [i for i in range(9)], 3, 2)
        tester(self, res)

    def test_smp(self):
        res = gen_starmap(test, [(i,) for i in range(9)], 3, 2)
        tester(self, res)
        res = gen_starmap(test, [(i,) for i in range(9)], 3)
        tester(self, res)

    def test_gmp(self):
        res = ChainIter([i for i in range(9)]).gen_map(test, 2, 3)
        tester(self, res)
        res = ChainIter([i for i in range(9)]).gen_map(test, 3)
        tester(self, res)

    def test_gsmp(self):
        res = ChainIter([(i, ) for i in range(9)]).gen_starmap(test, 2, 3)
        tester(self, res)
        res = ChainIter([(i, ) for i in range(9)]).gen_starmap(test, 3)
        tester(self, res)

def speed_test():
    print('first')
    current = time()
    ChainIter(range(10000)).map(normal_func).calc()
    print(time() - current)
    print('second')
    current = time()
    [normal_func(n) for n in range(10000)]
    print(time() - current)

def test(n):
    print(n, 'read')
    if n == 4:
        return 44
    yield 'step1'
    print(n, 'calc')
    sleep(0.1)
    if n == 6:
        raise OSError('hoge!')
    yield n + 10
    print(n, 'write')
    sleep(0.1)
    return n + 10

def test1(n):
    if n == 4:
        return 44
    yield 'step1'
    sleep(0.1)
    if n == 6:
        raise OSError('hoge!')
    yield n + 10
    sleep(0.1)
    return n + 10

result = [10, 11, 12, 13, 44, 15, OSError('hoge!'), 17, 18]
def tester(self, result1):
    for i, res, res1 in zip(range(9), result, result1):
        if i == 6:
            with self.assertRaises(OSError) as er:
                raise res1
        else:
            self.assertEqual(res, res1)
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

    
####################
if __name__ == '__main__':
    speed_test()
    async_speed_test()
    unittest.main()
