"""
Pipeline module for chainiter.
This is async and it must have Ninja speed!
"""
from typing import List, Any, Coroutine, Generator, Callable, Tuple, Iterable
from asyncio import new_event_loop, ensure_future, Future, sleep
from time import time
from threading import Thread
from queue import Queue
from multiprocessing import Pool
from functools import partial, reduce, wraps
from operator import add
from logging import (getLogger, Logger, NullHandler, StreamHandler,
                     INFO, WARNING, ERROR, CRITICAL)
from concurrent.futures import ThreadPoolExecutor
logger = getLogger('chainiter')
logger.addHandler(NullHandler())


def separate_list(data, sep) -> List[list]:
    length = len(data)
    ssep = int(length / sep)
    result = [[n for n in data[ind:ind + ssep]] for ind in range(0, length, ssep)]
    return result


class StoppableGenerator:
    """
    Stoppable generator object for async pipeline calculation.
    Python async function is the best usage of async programming in python.
    But, it is based on python itself, and so pure python pipeline is difficult.
    """
    def __init__(self, gen: Callable, args: Any, use_error: bool = True):
        self.args = args
        self.gen = gen(args)
        self.result = None
        self.use_error = use_error
        self.steps = 0

    def __next__(self):
        start_time = time()
        bubble = False
        try:
            res = next(self.gen)
            self.steps += 1
        except StopIteration as si:
            self.result = si.value
            self.steps = -1
        except BaseException as si:
            self.result = si if self.use_error else None
            self.steps = -1
            bubble = True
        finally:
            self.time = time() - start_time
            if bubble:
                return False


class ThreadPipeMap():
    """
    Low level map function object for pipeline calculation.
    It based not on async functions but on generators.
    It involves error processing.

    It is an iterator, but it is not used as an iterator.
    Just because iterator is useful to write this object.
    """
    def __init__(self, use_error: bool = True) -> None:
        """
        thread: int
            Depth of pipeline.
        use_error: bool
            Whether use error objects or not.
            If true, it returns list which involves error objects.
            If not, errors becomes None.
        """
        self.bubbles_num = 0
        self.use_error = use_error
    def map(self, func: Callable, args: Iterable) -> 'ThreadPipeMap':
        """
        Map function method.
        It returns itself.
        """
        self.gens = [StoppableGenerator(*g, self.use_error)
                     for g in ((func, a) for a in args)]
        self.cur_unit: int = 0
        return self

    def starmap(self, func: Callable,
                args: Iterable[Tuple[Any]]) -> 'ThreadPipeMap':
        """
        Starmap function method.
        It returns itself.
        """
        self.gens = [StoppableGenerator(*g, self.use_error)
                     for g in ((func, *a) for a in args)]
        # Current units
        self.cur_unit = 0
        return self

    def __iter__(self):
        """
        __iter__ method, but it is not reccomended to use it by your self.
        It is just a tool to process pipeline.
        """
        self.cycle = 0
        self.doing = []
        self.times = []
        return self

    def __next__(self):
        """
        __next__ method, but it is not recommended to use it by your self.
        It is just a tool to process pipeline.
        """
        self.cycle += 1
        if len(self.gens) != 0:
            self.doing += [self.gens.pop(0)]
        self.doing = [doing for doing in self.doing if doing.steps != -1]
        if len(self.doing) == 0:
            raise StopIteration
        with ThreadPoolExecutor(len(self.doing)) as exe:
            result = exe.map(next, self.doing)
            if len(self.times) != 0 and not all(result):
                self.times = [res.time for res in result]
                max_index = times.index(max(times))
                if result.index(False) < max_index and len(self.gens) != 0:
                    exe.map(next, doing[:max_index] + self.gens.pop(0))

    def get(self):
        """
        Get result of map and starmap.
        This is lazy but not asyncronious method.
        """
        tuple(self)
        return [self.gens[n].result for n in range(len(self.gens))]



def _tmap(gen, args, use_error: bool = True) -> list:
    """
    Low level thread based map function.
    It is based on ThreadPipeMap object.
    """
    return ThreadPipeMap(use_error).map(gen, args).get()
def gen_map(gen, args, processes=1, use_error: bool = True) -> list:
    """
    Generator based map function.
    It is based on ThreadPipeMap object.
    
    gen: Generator
        Generator to use pipeline.
    args: tuple
        Tuple or list like objects to use as a arguments of generators.
    threads: int
        Number of threads. This is based on depth of pipeline.
    processes: int
        Number of processes.
        If not 1, multi processing will be run multi processing.
    """
    if processes == 1:
        return _tmap(gen, args, use_error)
    args = separate_list(args, processes)
    with Pool(processes) as pool:
        result = pool.map_async(
            partial(_tmap, gen, use_error=use_error),
            args).get()
    return reduce(add, result)

def _tstarmap(gen, args, use_error: bool = True) -> list:
    """
    Low level thread based starmap function.
    It is based on ThreadPipeMap object.
    """
    return ThreadPipeMap(use_error).starmap(gen, args).get()

def gen_starmap(gen, args, processes=1, use_error: bool = True) -> list:
    """
    Generator based map function.
    It is based on ThreadPipeMap object.
    
    gen: Generator
        Generator to use pipeline.
    args: tuple
        Tuple or list like objects to use as a arguments of generators.
    threads: int
        Number of threads. This is based on depth of pipeline.
    processes: int
        Number of processes.
        If not 1, multi processing will be run multi processing.
    """
    if processes == 1:
        return _tstarmap(gen, args, use_error)
    args = separate_list(args, processes)
    with Pool(processes) as pool:
        result = pool.starmap_async(
            _tstarmap, ((gen, ar, use_error) for ar in args)).get()
    return reduce(add, result)
