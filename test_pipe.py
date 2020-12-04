from asyncio import sleep, ensure_future, Future, wait, get_event_loop
import time
from chainiter.pipline import ThreadPipeMap, tmap, tstarmap, tpmap, tpstarmap, separate_list
from logging import basicConfig, INFO
from functools import partial
from unittest import TestCase, main
from multiprocessing import Pool
basicConfig(level=INFO)



def test(n):
    if n == 4:
        return 44
    yield 'step1'
    time.sleep(0.1)
    if n == 6:
        raise OSError('hoge!')
    yield n + 10
    time.sleep(0.1)
    return n + 10

def test1(n):
    if n == 4:
        return 44
    yield 'step1'
    time.sleep(0.1)
    if n == 6:
        raise OSError('hoge!')
    yield n + 10
    time.sleep(0.1)
    return n + 10


result = [10, 11, 12, 13, 44, 15, OSError('hoge!'), 17, 18]
def tester(self, result1):
    for i, res, res1 in zip(range(9), result, result1):
        if i == 6:
            with self.assertRaises(OSError) as er:
                raise res1
        else:
            self.assertEqual(res, res1)


class Test(TestCase):
    def test_pipe_starmap(self):
        tester(self, ThreadPipeMap(3).starmap(test, [(i,) for i in range(9)]).get())

    def test_ThreadPipeMap(self):
        tester(self, ThreadPipeMap(3).map(test, [i for i in range(9)]).get())

    def test_mp(self):
        res = tpmap(test, [i for i in range(9)], 3, 2)
        tester(self, res)
        res = tmap(test, [i for i in range(9)], 3)
        tester(self, res)

    def test_smp(self):
        res = tpstarmap(test, [(i,) for i in range(9)], 3, 2)
        tester(self, res)
        res = tstarmap(test, [(i,) for i in range(9)], 3)
        tester(self, res)

main()
# for ii in i:
#     if isinstance(ii, BaseException):
#         raise ii
#     print(ii)
