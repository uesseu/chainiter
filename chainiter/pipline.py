"""
Pipeline module for chainiter.
This is async and it must have Ninja speed!
"""
from typing import List, Any, Coroutine, Generator, Callable, Tuple, Iterable
from asyncio import new_event_loop, ensure_future, Future, sleep
import time
from threading import Thread
from queue import Queue
from multiprocessing import Pool
from functools import partial, reduce, wraps
from operator import add
from logging import (getLogger, Logger, NullHandler, StreamHandler,
                     INFO, WARNING, ERROR, CRITICAL)
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
        self.gen = gen(args)
        self.result = None
        self.end = False
        self.bubble = False
        self.use_error = use_error

    def __next__(self):
        if self.end:
            return False
        try:
            res = next(self.gen)
        except StopIteration as si:
            self.end = True
            self.result = si.value
        except BaseException as si:
            self.end = True
            self.bubble = True
            self.result = si if self.use_error else None

def isnot_ended(sg: StoppableGenerator) -> bool: return not sg.end
def isbubble(sg: StoppableGenerator) -> bool: return sg.bubble
class ThreadPipeMap():
    """
    Map function object for pipeline calculation.
    It based not on async functions but on generators.
    It involves error processing.

    It is an iterator, but it is not used as an iterator.
    Just because iterator is useful to write this object.
    """
    def __init__(self, sep: int, use_error: bool = True) -> None:
        """
        sep: int
            Depth of pipeline.
        use_error: bool
            Whether use error objects or not.
            If true, it returns list which involves error objects.
            If not, errors becomes None.
        """
        self.sep = sep
        self.bubbles_num = 0
        self.use_error = use_error
    def map(self, func: Callable, args: Iterable) -> 'ThreadPipeMap':
        """
        Map function method.
        It returns itself.
        """
        self.gens = [StoppableGenerator(*g, self.use_error)
                     for g in ((func, a) for a in args)]
        self.cursep: int = 0
        return self
    def starmap(self, func: Callable,
                args: Iterable[Tuple[Any]]) -> 'ThreadPipeMap':
        """
        Starmap function method.
        It returns itself.
        """
        self.gens = [StoppableGenerator(*g, self.use_error)
                     for g in ((func, *a) for a in args)]
        self.cursep = 0
        return self

    def __iter__(self):
        """
        __iter__ method, but it is not reccomended to use it by your self.
        It is just a tool to process pipeline.
        """
        return self

    def __next__(self):
        """
        __next__ method, but it is not recommended to use it by your self.
        It is just a tool to process pipeline.
        """
        if self.sep != self.cursep:
            self.cursep += 1
        alive_gens = list(filter(isnot_ended, self.gens))
        if len(alive_gens) == 0:
            raise StopIteration
        alive_num = len(alive_gens)
        num_sep: int = alive_num if self.cursep > alive_num else self.cursep
        num_sep = num_sep - self.bubbles_num
        threads = tuple(Thread(target=next, args=(gen,))
                        for gen in alive_gens[0: num_sep])
        tuple(t.start() for t in threads)
        tuple(t.join() for t in threads)
        self.bubbles_num = max(len(list(filter(isbubble, self.gens))) - 1, 0)
        if self.bubbles_num == self.cursep:
            self.bubbles_num = 0

    def get(self):
        """
        Get result of map and starmap.
        This is lazy but not asyncronious method.
        """
        tuple(self)
        return [self.gens[n].result for n in range(len(self.gens))]

class PipeMap():
    """
    Map function object for pipeline calculation.
    It based not on async functions but on generators.
    It involves error processing.
    """
    def __init__(self, sep: int, use_error: bool = True) -> None:
        """
        sep: int
            Depth of pipeline.
        use_error: bool
            Whether use error objects or not.
            If true, it returns list which involves error objects.
            If not, errors becomes None.
        """
        self.sep = sep
        self.bubbles_num = 0
        self.use_error = use_error
        self._num = 0
        self._tpm: ThreadPipeMap

    def __iter__(self):
        self._num = 0
        self.data = self._tpm.get()
        self._len = len(self.data)
        return self

    def __next__(self):
        if self._num + 1 > self._len:
            raise StopIteration()
        result = self.data[self._num]
        self._num += 1
        return result

    def map(self, func: Callable, args: Iterable) -> 'PipeMap':
        """
        Map function method.
        It returns itself.
        """
        self._tpm = ThreadPipeMap(self.sep).map(func, args)
        return self

    def starmap(self, func: Callable,
                args: Iterable[Tuple[Any]]) -> 'PipeMap':
        """
        Starmap function method.
        It returns itself.
        """
        self._tpm = ThreadPipeMap(self.sep).starmap(func, args)
        return self


def _tmap(gen, args, threads, use_error: bool = True) -> list:
    """
    Thread based map function.
    It is based on ThreadPipeMap object.
    """
    return ThreadPipeMap(threads, use_error).map(gen, args).get()

def _tstarmap(gen, args, threads, use_error: bool = True) -> list:
    """
    Thread based starmap function.
    It is based on ThreadPipeMap object.
    """
    return ThreadPipeMap(threads, use_error).starmap(gen, args).get()

def tmap(gen, args, threads, use_error: bool = True) -> list:
    """
    Thread based map function.
    It is based on ThreadPipeMap object.
    """
    return PipeMap(threads, use_error).map(gen, args)

def tstarmap(gen, args, threads, use_error: bool = True) -> list:
    """
    Thread based starmap function.
    It is based on ThreadPipeMap object.
    """
    return PipeMap(threads, use_error).starmap(gen, args)

def tpmap(gen, args, threads, processes, use_error: bool = True) -> list:
    args = separate_list(args, processes)
    with Pool(processes) as pool:
        result = pool.map_async(
            partial(_tmap, gen, threads=threads, use_error=use_error),
            args).get()
    return reduce(add, result)

def tpstarmap(gen, args, threads, processes, use_error: bool = True) -> list:
    args = separate_list(args, processes)
    with Pool(processes) as pool:
        result = pool.map_async(
            partial(_tstarmap, gen, threads=threads, use_error=use_error),
            args).get()
    return reduce(add, result)
