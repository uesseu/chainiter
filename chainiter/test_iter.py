from chainiter import ChainIter, future, ProgressBar, curry, logger, start_async, run_async, run_coroutine
from doctest import testmod
from time import sleep, time
import asyncio
from logging import getLogger, basicConfig, INFO, WARNING, StreamHandler
from itertools import product
basicConfig(level=INFO)
# logger.addHandler(StreamHandler())


####################
# Curry test
####################
@curry(2)
def mm(x, y):
    return x * y
print(mm(9)(8))
####################

####################
# Future test
####################
def sleep_test(x: int) -> int:
    print(x)
    sleep(1)
    return x * 2
async def _future_test():
    print('sleep')
    a = start_async(asyncio.sleep, 1)
    b = start_async(asyncio.sleep, 2)
    c = start_async(asyncio.sleep, 3)
    d = start_async(asyncio.sleep, 4)
    result = await asyncio.gather(a, b, c, d)

def future_test():
    print(run_coroutine(_future_test()))
    print('slept')

####################

async def itest(x: int) -> int: return x * 2

def multest(x: int) -> int: return x * 2
@future
async def hoge(x: int) -> int:
    ''' pp '''
    return x * 2

async def test(x: int) -> int:
    # print(hoge.__doc__)
    return await hoge(x)

def test3(x: int, y: int) -> int: return x * y
def test2(x: int) -> int: return x * 2
def titest(x: int) -> int:
    sleep(0.05)
    return x * 2

async def async_titest(x: int) -> int:
    asyncio.sleep(0.05)
    await asyncio.sleep(0.05)
    return x * 2

####################
# Normal test
####################
ChainIter([1, 2, 3]).map(test2).calc()
ChainIter([(1, 2), (3, 5)]).starmap(test3).calc()
####################
####################
# Speed test
####################
def speed_test():
    current = time()
    ChainIter(range(1000000)).map(test2).calc()
    print(time() - current)

    current = time()
    [test2(n) for n in range(1000000)]
    print(time() - current)
####################

####################
# Async speed test
####################
def async_speed_test():
    current = time()
    ChainIter(range(100)).async_map(async_titest).calc()
    print(time() - current)

    current = time()
    ChainIter(range(100)).map(titest).calc()
    print(time() - current)

    current = time()
    ChainIter(range(100)).async_map(async_titest, 5).calc()
    print(time() - current)

    current = time()
    ChainIter(range(100)).map(titest, 5).calc()
    print(time() - current)
####################

def api_test():
    print(ChainIter(range(10)) > 5)
    bar = True#ProgressBar()
    print(ChainIter([2, 3, 4], bar=bar).async_map(test).map(test2)[0])
    print(ChainIter(range(30), bar=bar).map(titest).map(test2).calc())
    print(ChainIter([1, 2]))
    print(ChainIter([5, 3]).async_map(itest, 2).map(test2, 2).get())
    for n in ChainIter(range(40), bar=bar).async_map(test).map(test2):
        sleep(0.05)
    print(ChainIter([2, 3]).async_map(test, 2).map(test2, 2).get(tuple))
    if isinstance(bar, ProgressBar):
        bar.arrow = '8'
    coco = ChainIter(range(50), bar=bar)
    coco.map(titest, 2).calc().get(tuple)
    ho = ChainIter([2, 3]).async_map(test).map(test2).get()
    print(ho[0])
    print(ho[1])


####################
# Async starmap test
####################
async def async_test3(x: int, y: int) -> int: return x * y
ChainIter(product([1, 2], [3, 4])).async_starmap(async_test3, 2).calc().print()
async def async_ptest3(x: int, y: int, z: int) -> int: return x * y + z
ChainIter([1, 2]).async_pmap(2)(async_test3, y=7).calc().print()
async def async_ptest4(x: int, y: int, z: int, zz: int) -> int:
    print(x)
    return x * y + z - zz
ChainIter([1, 2]).zip([3, 4]).calc().print().async_pstarmap(2)(async_ptest4, 7, 1).calc().print()
ChainIter([1, 2]).async_pmap(2)(async_ptest3, 7, 1).calc().print()
####################
speed_test()
async_speed_test()
future_test()
api_test()
