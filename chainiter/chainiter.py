"""
Ninja speed iterator package.
This is an iterator like Array object of Node.js.
Chainable fast iterators can be used.
Furturemore, generator based pipeline can be used.
"""

# Chainiter is built on objects.
# ChainIterPrivate
# ChainIterBase
# ChainIterNormal
# ChainIterAsync
# 
# Object hierarchie is...
# Private -> Base
# Base -> Normal
# Base -> Async
# Base -> Gen
# Normal + Async + Gen-> ChainIter
from asyncio import new_event_loop, ensure_future, Future
from typing import (Any, Callable, Iterable, cast, Coroutine,
                    Union, Sized, Optional, TypeVar, Coroutine,
                    Tuple, Generator, Iterator)
from itertools import starmap, product
from collections import namedtuple
from multiprocessing import Pool
from doctest import testmod
from functools import reduce, wraps, partial
from logging import (getLogger, Logger, NullHandler, StreamHandler,
                     INFO, WARNING, ERROR, CRITICAL)
from inspect import _empty, signature
import os
from contextlib import redirect_stdout
import time
from threading import Thread
from queue import Queue
import warnings
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from .pipeline import gen_map, gen_starmap
logger = getLogger('chainiter')
logger.addHandler(NullHandler())


class ChainIterMode(Enum):
    SINGLE = 'single'
    THREAD = 'thread'
    PROCESS = 'process'
    BOTH = 3


mode = ChainIterMode

def compose(*funcs: Callable) -> Callable:
    """
    Just a compose function
    >>> def multest(x): return x * 2
    >>> compose(multest, multest)(4)
    16
    """
    def composed(*args: Any) -> Any:
        return reduce(lambda x, y: y(x), (funcs[0](*args),) + funcs[1:])
    return composed


def _curry_one(func: Callable) -> Callable:
    """
    >>> def multest2(x, y): return x * y
    >>> fnc = _curry_one(multest2)
    >>> fnc(2)(3)
    6
    """
    def wrap(*args: Any, **kwargs: Any) -> Any:
        return partial(func, *args, **kwargs)
    return wrap


def curry(num_of_args: Optional[int] = None) -> Callable:
    """
    Just a curry function.
    >>> def multest2(x, y): return x * y
    >>> fnc = curry(2)(multest2)
    >>> fnc(2)(3)
    6
    >>> def multest3(x, y, z): return x * y * z
    >>> fnc = curry()(multest3)
    >>> fnc(2)(3)(4)
    24
    """
    def curry_wrap(func: Callable) -> Callable:
        wr = wraps(func)
        length_of = compose(filter, list, len)
        if num_of_args:
            num = num_of_args
        else:
            def is_empty(x: Any) -> bool: return x.default is _empty
            num = length_of(is_empty, signature(func).parameters.values())
        for n in range(num - 1):
            func = _curry_one(func)
        return wr(func)
    return curry_wrap


def write_info(func: Callable, chunk: int = 1,
               logger: Logger = logger, mode: str = 'single') -> None:
    """
    Log displayer of chainiter.
    """
    if hasattr(func, '__name__'):
        logger.info(' '.join(('Running', str(func.__name__))))
    else:
        logger.info('Running something without name.')
    if chunk > 1:
        logger.info(f'Computing by {chunk} {mode}!')


class ProgressBar:
    """
    A configuration object of progressbar for chainiter.
    There are some parameters of progress bar.

    percent: Percent of progress bar.
    bar: Body of bar.
    arrow: Top of bar.
    space: Un reached space of bar.
    div: The end of space.
    cycle: Cycle style counter.
    epoch_time: Time spent by one epoch.
    speed: Speed of processing.

    Default value is below.

    self.bar_str = '\r{percent}%[{bar}{arrow}{space}]{div}'
    self.cycle_token = ('-', '\\', '|', '/')
    self.cycle_str = '\r[{cycle}]'
    self.stat_str = ' | {epoch_time:.2g}sec/epoch | Speed: {speed:.2g}/sec'
    self.progress = self.bar_str + self.stat_str
    self.cycle = self.cycle_str + self.stat_str
    self.bar = '='
    self.space = ' '
    self.arrow = '>'
    """

    def __init__(self) -> None:
        self.bar_str = '\r{percent}%[{bar}{arrow}{space}]{div}'
        self.cycle_token = ('-', '\\', '|', '/')
        self.cycle_str = '\r[{cycle}]'
        self.stat_str = ' | {epoch_time:.2g}sec/epoch | Speed: {speed:.2g}/sec'
        self.progress = self.bar_str + self.stat_str
        self.cycle = self.cycle_str + self.stat_str
        self.bar = '='
        self.space = ' '
        self.arrow = '>'


default_progressbar = ProgressBar()


def run_coroutine(col: Coroutine) -> Any:
    loop = new_event_loop()
    result = loop.run_until_complete(col)
    loop.close()
    return result

def start_async(func: Callable, *args, **kwargs) -> Future:
    """
    Start async function instantly.
    """
    async def wrap():
        return func(*args, **kwargs)
    return ensure_future(wrap())

def future(func: Callable) -> Callable:
    """
    Let coroutine return future object.
    It can be used as decorator.
    """
    @wraps(func)
    def wrap(*args: Any, **kwargs: Any) -> Future:
        return ensure_future(func(*args, **kwargs))
    return wrap


def run_async(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Assemble coroutine and run.

    For example...

    from chainiter import future, run_async

    @future
    async def hoge():
        return 'fuga'
    fuga = run_async(hoge())
    """
    result = func(*args, **kwargs)
    if isinstance(result, Coroutine):
        loop = new_event_loop()
        result = loop.run_until_complete(result)
        loop.close()
    return result

def as_sync(func):
    """
    Make an async function a normal function.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        loop = new_event_loop()
        result = loop.run_until_complete(func(*args, **kwargs))
        loop.close()
        return result
    return wrap

def as_sync(func):
    """
    Make an async function a normal function.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        loop = new_event_loop()
        result = loop.run_until_complete(func(*args, **kwargs))
        loop.close()
        return result
    return wrap


def make_color(txt: str, num: int) -> str:
    color = '\033[9' + str(6 - num) + 'm'
    reset = '\033[0m'
    return color + txt + reset

class ChainBase:
    def __init__(self, data: Union[list, Iterable],
                 indexable: bool = False, max_num: int = 0,
                 bar: Union[ProgressBar, None, bool] = None):
        """
        Parameters
        ----------
        data: Iterable
            It need not to be indexable.
        indexable: bool
            If data is indexable, indexable should be True.
        max_num: int
            Length of the iterator.
        bar: Optional[ProgressBar]
            If ProgressBar object, it will display progressbar
            when for statement or .calc method.
        """
        if not hasattr(data, '__iter__'):
            TypeError('It is not iterator')
        self.data = data
        self.indexable = indexable
        self._num = 0  # Iterator needs number.
        self.bar: Union[ProgressBar, None, bool]
        if hasattr(data, '__len__'):
            self._max = len(cast(Sized, data))
        else:
            self._max = max_num
        if isinstance(bar, bool) and bar:
            self.bar = ProgressBar()
        else:
            self.bar = bar
        self._bar_len = 30

    def calc(self) -> 'ChainIter':
        """
        ChainIter.data may be list, map, filter and so on.
        This method translate it to list.
        If you do not run in parallel, it can print progress bar.

        Returns
        ----------
        ChainIter object with result.
        """
        if isinstance(self.bar, ProgressBar) and not self.indexable:
            res = []
            start_time = current_time = time.time()
            for n, v in enumerate(self.data):
                res.append(v)
                prev_time = current_time
                current_time = time.time()
                epoch_time = current_time - prev_time
                if self._max != 0:
                    bar_num = int((n + 1) / self._max * self._bar_len)
                    percent = int(100 * (n + 1) / self._max)
                    bar = self.bar.bar * bar_num
                    print(self.bar.progress.format(
                        percent=percent,
                        bar=bar,
                        arrow=self.bar.arrow,
                        space=self.bar.space * (self._bar_len - bar_num),
                        div=' ' + str(n + 1) + '/' + str(self._max),
                        epoch_time=round(epoch_time, 3),
                        speed=round(1 / epoch_time, 3)
                        ), end='')
                else:
                    print(self.bar.cycle.format(
                        cycle=self.bar.cycle_token[n % 4],
                        div=' ' + str(n + 1) + '/' + str(self._max),
                        epoch_time=round(epoch_time, 3),
                        speed=round(1 / epoch_time, 3)
                        ), end='')
            print('\nComplete in {sec} sec!'.format(
                sec=round(time.time()-start_time, 3)))
            self.data = res
            return cast(ChainIter, self)
        else:
            self.data = list(self.data)
        return cast(ChainIter, self)

class ChainOperator(ChainBase):
    def __add__(self, item: Any) -> 'ChainIter':
        if not hasattr(self.data, 'append'):
            raise IndexError('Run ChainIter.calc().')
        cast(list, self.data).append(item)
        return cast(ChainIter, self)

    def __setitem__(self, key: Any, item: Any) -> None:
        if hasattr(self.data, '__setitem__'):
            cast(list, self.data)[key] = item
        raise IndexError('Item cannot be set. Run ChainIter.calc().')

    def __lt__(self, arg: Any) -> list:
        return ChainIter(map(lambda x: x < arg, self.data)).get()

    def __le__(self, arg: Any) -> list:
        return ChainIter(map(lambda x: x <= arg, self.data)).get()

    def __gt__(self, arg: Any) -> list:
        return ChainIter(map(lambda x: x > arg, self.data)).get()

    def __ge__(self, arg: Any) -> list:
        return ChainIter(map(lambda x: x >= arg, self.data)).get()


class ChainPrivate(ChainOperator):

    def __reversed__(self) -> Iterable:
        if hasattr(self.data, '__reversed__'):
            return cast(list, self.data).__reversed__()
        raise IndexError('Not reversible')

    def _has_index(self) -> bool:
        """
        Return whether it is indexable or not.
        """
        return True if self.indexable else hasattr(self.data, '__getitem__')

    def __len__(self) -> int:
        if not self.indexable:
            self.calc()
            self.indexable = True
        return len(cast(list, self.data))

    def __repr__(self) -> str:
        return 'ChainIter[{}]'.format(str(self.data))

    def __str__(self) -> str:
        return 'ChainIter[{}]'.format(str(self.data))

    def __getitem__(self, num: int) -> Any:
        if self._has_index():
            return cast(list, self.data)[num]
        self.data = tuple(self.data)
        return self.data[num]

    def __iter__(self) -> 'ChainIter':
        bar = self.bar
        self.bar = None
        self.calc()
        self.bar = bar
        self._max = len(cast(list, self.data))
        self._num = 0
        self.start_time = self.current_time = time.time()
        return cast(ChainIter, self)

    def __next__(self) -> Any:
        if isinstance(self.bar, ProgressBar):
            self._prev_time = self.current_time
            self.current_time = time.time()
            epoch_time = self.current_time - self._prev_time + 1e-15
            if self._max != 0:
                bar_num = int((self._num + 1) / self._max * self._bar_len)
                print(self.bar.progress.format(
                    percent=int(100 * (self._num) / self._max),
                    bar=self.bar.bar * bar_num,
                    arrow=self.bar.arrow,
                    space=self.bar.space * (self._bar_len - bar_num),
                    div=' ' + str(self._num) + '/' + str(self._max),
                    epoch_time=round(epoch_time, 3),
                    speed=round(1 / epoch_time, 3)
                    ), end='')
            else:
                print(self.bar.cycle.format(
                    cycle=self.bar.cycle_token[self._num % 4],
                    div=' ' + str(self._num) + '/' + str(self._max),
                    epoch_time=round(epoch_time, 3),
                    speed=round(1 / epoch_time, 3)
                    ), end='')
        if self._num == self._max:
            if self.bar:
                print('\nComplete in {sec} sec!'.format(
                    sec=round(time.time() - self.start_time, 3)))
            self._num = 0
            raise StopIteration
        result = self.__getitem__(self._num)
        self._num += 1
        return result

class ChainIterBase(ChainPrivate):
    def append(self, item: Any) -> 'ChainIter':
        if not isinstance(self.data, list):
            self.calc()
        cast(list, self.data).append(item)
        return cast(ChainIter, self)

    def get(self, kind: type = list) -> Any:
        """
        Get data as list.

        Parameters
        ----------
        kind: Callable
            If you want to convert to object which is not list,
            you can set it. For example, tuple, dqueue, and so on.
        """
        return kind(self.data)


class ChainIterNormal(ChainIterBase):

    def zip(self, *args: Iterable) -> 'ChainIter':
        """
        Simple chainable zip function.
        It kills progress bar.

        Parameters
        ----------
        *args: Iterators to zip.

        Returns
        ----------
        Result of func(*ChainIter, *args, **kwargs)
        """
        return ChainIter(zip(self.data, *args), False, 0, self.bar)

    def reduce(self, func: Callable, logger: Logger = logger) -> Any:
        """
        Simple reduce function.

        Parameters
        ----------
        func: Callable
        logger: logging.Logger
            Your favorite logger.

        Returns
        ----------
        Result of reduce.
        """
        write_info(func, 1, logger)
        return reduce(func, self.data)

    def map(self, func: Callable, process: int = 1,
            thread: int = 1,
            timeout: Optional[float] = None,
            logger: Logger = logger) -> 'ChainIter':
        """
        Chainable map.

        Parameters
        ----------
        func: Callable
            Function to run.
        chunk: int
            Number of cpu cores.
            If it is larger than 1, multiprocessing based on
            multiprocessing.Pool will run.
            And so, If func cannot be lambda or coroutine if
            it is larger than 1.
        timeout: Optional[float] = None
            Time to stop parallel computing.
        logger: logging.Logger
            Your favorite logger.
        Returns
        ---------
        ChainIter with result

        >>> ChainIter([5, 6]).map(lambda x: x * 2).get()
        [10, 12]
        """
        _func = partial(run_async, func)
        if process == 1 and thread == 1:
            write_info(func, 1, logger, 'single')
            return ChainIter(map(_func, self.data),
                             False, self._max, self.bar)
        elif process != 1:
            write_info(func, process, logger, 'process')
            with Pool(process) as pool:
                result = pool.map_async(_func, self.data).get(timeout)
            return ChainIter(result, True, self._max, self.bar)
        else:
            write_info(func, thread, logger, 'thread')
            with ThreadPoolExecutor(thread) as exe:
                result = exe.map(_func, self.data)
            return ChainIter(result, True, self._max, self.bar)

    def starmap(self, func: Callable, process: int = 1, thread: int = 1,
                timeout: Optional[float] = None,
                logger: Logger = logger) -> 'ChainIter':
        """
        Chainable starmap.
        In this case, ChainIter.data must be Iterator of iterable objects.

        Parameters
        ----------
        func: Callable
            Function to run.
        chunk: int
            Number of cpu cores.
            If it is larger than 1, multiprocessing based on
            multiprocessing.Pool will be run.
            And so, If func cannot be lambda or coroutine if
            it is larger than 1.
        timeout: Optional[float] = None
            Time to stop parallel computing.
        logger: logging.Logger
            Your favorite logger.
        Returns
        ---------
        ChainIter with result
        >>> def multest2(x, y): return x * y
        >>> ChainIter([5, 6]).zip([2, 3]).starmap(multest2).get()
        [10, 18]
        """
        _func = partial(run_async, func)
        if process == 1 and thread == 1:
            write_info(func, 1, logger, 'single')
            return ChainIter(starmap(_func, self.data),
                             False, self._max, self.bar)
        elif process != 1:
            write_info(func, process, logger, 'single')
            with Pool(process) as pool:
                result = pool.starmap_async(_func, self.data).get(timeout)
            return ChainIter(result, True, self._max, self.bar)
        else:
            write_info(func, thread, logger, 'single')
            result = thread_starmap(_func, self.data)
            return ChainIter(result, True, self._max, self.bar)

    def filter(self, func: Callable, logger: Logger = logger) -> 'ChainIter':
        """
        Simple filter function.
        It kills progress bar.

        Parameters
        ----------
        func: Callable
        logger: logging.Logger
            Your favorite logger.
        """
        write_info(func, 1, logger)
        return ChainIter(filter(func, self.data), False, 0, self.bar)

class ChainIterGenerator(ChainIterBase):
    def gen_map(self, gen, processes=1, use_error: bool = True) -> 'ChainIter':
        return ChainIter(
            gen_map(gen, self.data, processes, use_error),
            True, self._max, self.bar)

    def gen_starmap(self, gen, processes=1, use_error: bool = True) -> 'ChainIter':
        return ChainIter(
            gen_starmap(gen, self.data, processes, use_error),
            True, self._max, self.bar)

class ChainIterAsync(ChainIterBase):
    def async_map(self, func: Callable, process: int = 1, thread: int = 1,
                  timeout: Optional[float] = None,
                  logger: Logger = logger) -> 'ChainIter':
        """
        Chainable map of coroutine, for example, async def function.

        Parameters
        ----------
        func: Callable
            Function to run.
        chunk: int
            Number of cpu cores.
            If it is larger than 1, multiprocessing based on
            multiprocessing.Pool will be run.
            And so, If func cannot be lambda or coroutine if
            it is larger than 1.
        timeout: Optional[float] = None
            Time to stop parallel computing.
        logger: logging.Logger
            Your favorite logger.
        Returns
        ---------
        ChainIter with result
        >>> async def multest(x): return x * 2
        >>> ChainIter([1, 2]).async_map(multest).get()
        [2, 4]
        """
        if process == 1 and thread == 1:
            write_info(func, 1, logger, 'single')
            return ChainIter(
                starmap(run_async, ((func, a) for a in self.data)),
                False, self._max, self.bar)
        elif process != 1:
            write_info(func, process, logger, 'process')
            with Pool(process) as pool:
                result = pool.starmap_async(run_async,
                    ((func, a) for a in self.data)).get(timeout)
            return ChainIter(result, True, self._max, self.bar)
        else:
            write_info(func, thread, logger, 'thread')
            result = thread_starmap(run_async,
                                    ((func, a) for a in self.data), thread)
            return ChainIter(result, True, self._max, self.bar)

    def async_starmap(self, func: Callable, process: int = 1, thread: int = 1,
                      timeout: Optional[float] = None,
                      logger: Logger = logger) -> 'ChainIter':
        """
        Chainable starmap of coroutine, for example, async def function.

        Parameters
        ----------
        func: Callable
            Function to run.
        chunk: int
            Number of cpu cores.
            If it is larger than 1, multiprocessing based on
            multiprocessing.Pool will be run.
            And so, If func cannot be lambda or coroutine if
            it is larger than 1.
        timeout: Optional[float] = None
            Time to stop parallel computing.
        logger: logging.Logger
            Your favorite logger.
        Returns
        ---------
        ChainIter with result
        >>> async def multest(x, y): return x * y
        >>> ChainIter(zip([5, 6], [1, 3])).async_starmap(multest).get()
        [5, 18]
        """
        if process == 1 and thread == 1:
            write_info(func, 1, logger, 'single')
            return ChainIter(
                starmap(run_async, ((func, *a) for a in self.data)),
                False, self._max, self.bar)
        elif process != 1:
            write_info(func, process, logger, 'single')
            with Pool(process) as pool:
                result = pool.starmap_async(
                    run_async, ((func, *a) for a in self.data)).get(timeout)
            return ChainIter(result, True, self._max, self.bar)
        else:
            write_info(func, thread, logger, 'single')
            result = thread_starmap(run_async,
                           ((func, *a) for a in self.data), thread)
            return ChainIter(result, True, self._max, self.bar)


class ChainMisc(ChainBase):
    def print(self) -> 'ChainIter':
        """
        Just print the content.
        """
        print(self.data)
        return cast(ChainIter, self)

    def log(self, logger: Logger = logger, level: int = INFO) -> 'ChainIter':
        """
        Just print the content.
        """
        if level == INFO:
            logger.info(self.data)
        elif level == WARNING:
            logger.warning(self.data)
        elif level == ERROR:
            logger.error(self.data)
        elif level == CRITICAL:
            logger.critical(self.data)
        return cast(ChainIter, self)

    def print_len(self) -> 'ChainIter':
        """
        Just print length of the content.
        """
        print(len(list(self.data)))
        return cast(ChainIter, self)

    def arg(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Use ChainIter object as argument.
        It is same as func(*ChainIter, *args, **kwargs)

        Parameters
        ----------
        func: Callable

        Returns
        ----------
        Result of func(*ChainIter, *args, **kwargs)
        >>> ChainIter([5, 6]).arg(sum)
        11
        """
        return func(tuple(self.data), *args, **kwargs)

    def stararg(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Use ChainIter object as argument.
        It is same as func(*tuple(ChainIter), *args, **kwargs)

        Parameters
        ----------
        func: Callable

        Returns
        ----------
        ChainIter object
        >>> ChainIter([5, 6]).stararg(lambda x, y: x * y)
        30
        """
        return func(*tuple(self.data), *args, **kwargs)


class ChainIter(ChainIterNormal, ChainMisc, ChainIterAsync, ChainIterGenerator):
    """
    Iterator which can used by method chain like Arry of node.js.
    Multi processing and asyncio can run.
    """
    def __init__(self, data: Union[list, Iterable],
                 indexable: bool = False, max_num: int = 0,
                 bar: Union[ProgressBar, None, bool] = None):
        super(ChainIter, self).__init__(
            data, indexable, max_num, bar)


def _tmp_thread(args: Any) -> Any: return args[0](*args[1:])
def thread_starmap(func: Callable, args: Union[list, Iterable],
                   chunk: int = 1) -> Iterator:
    with ThreadPoolExecutor(chunk) as exe:
        result = exe.map(_tmp_thread, ((func, *a) for a in args))
    return result


if __name__ == '__main__':
    testmod()
