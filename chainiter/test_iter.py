from chainiter import ChainIter, future, ProgressBar, curry
from doctest import testmod
from time import sleep, time
from logging import basicConfig, INFO
from itertools import product
# basicConfig(level=INFO)

####################
# Curry test
####################
@curry(2)
def mm(x, y):
    return x * y
print(mm(9)(8))
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


####################
# Speed test
####################
current = time()
ChainIter(range(1000000)).map(test2).calc()
print(time() - current)

current = time()
[test2(n) for n in range(1000000)]
print(time() - current)
####################


print(ChainIter(range(10)) > 5)
bar = ProgressBar()
print(ChainIter([2, 3, 4], bar=bar).async_map(test).map(test2)[0])
print(ChainIter(range(30), bar=bar).map(titest).map(test2).calc())
print(ChainIter([1, 2]))
print(ChainIter([5, 3]).async_map(itest, 2).map(test2, 2).get())
for n in ChainIter(range(40), bar=bar).async_map(test).map(test2):
    sleep(0.05)
print(ChainIter([2, 3]).async_map(test, 2).map(test2, 2).get(tuple))
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
####################