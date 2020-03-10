from chainiter import ChainIter, future
from doctest import testmod
from time import sleep, time
from logging import basicConfig, INFO
# basicConfig(level=INFO)

async def itest(x): return x * 2

def multest(x: int) -> int: return x * 2
@future
async def hoge(x: int) -> int:
    ''' pp '''
    return x * 2

async def test(x: int) -> int:
    # print(hoge.__doc__)
    return await hoge(x)

def test3(x, y): return x * y
def test2(x: int) -> int: return x * 2
def titest(x: int) -> int:
    sleep(0.05)
    return x * 2

current = time()
ChainIter(range(10000000)).map(test2).calc()
print(time() - current)

current = time()
[test2(n) for n in range(10000000)]
print(time() - current)


print(ChainIter([2, 3, 4], bar=True).async_map(test).map(test2)[0])
print(ChainIter(range(30), bar=True).map(titest).map(test2).calc())
print(ChainIter([1, 2]))
print(ChainIter([5, 3]).async_map(itest, 2).map(test2, 2)[0])
for n in ChainIter(range(40), bar=True).async_map(test).map(test2):
    sleep(0.05)
print(ChainIter([2, 3]).async_map(test, 2).map(test2, 2).get(tuple))
coco = ChainIter(range(50), bar=True)
coco.progressbar.arrow = '8'
coco.map(titest).calc().get(tuple)
ho = ChainIter([2, 3]).async_map(test).map(test2).get()
print(ho[0])
print(ho[1])
