from asyncio import new_event_loop, ensure_future, Future
from typing import (Any, Callable, Iterable, cast, Coroutine,
                    Iterator, List, Union, Sized)
from itertools import starmap
from multiprocessing import Pool
from doctest import testmod
from functools import reduce, wraps, partial
from logging import Logger, INFO, getLogger, basicConfig
import time


logger = getLogger('ChainIter')


class ProgressBar:
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
    loop = new_event_loop()
    result = loop.run_until_complete(func(*args, **kwargs))
    loop.close()
    return result


def make_color(txt: str, num: int) -> str:
    color = '\033[9' + str(6 - num) + 'm'
    reset = '\033[0m'
    return color + txt + reset


class ChainIter:
    """
    Iterator which can used by method chain like Arry of node.js.
    Multi processing and asyncio can run.
    """
    def __init__(self, data: Union[list, Iterable],
                 indexable: bool = False, max_num: int = 0, bar: bool = False,
                 progressbar: ProgressBar = default_progressbar):
        """
        Parameters
        ----------
        data: Iterable
            It need not to be indexable.
        indexable: bool
            If data is indexable, indexable should be True.
        max_num: int
            Length of the iterator.
        bar: bool
            Whether show progress bar or not.
            It is fancy, but may be slower.
            It cannot run with multiprocess.
        """
        self.data = data
        self.indexable = indexable
        self.num = 0  # Iterator needs number.
        if hasattr(data, '__len__'):
            self.max = len(cast(Sized, data))
        else:
            self.max = max_num
        self.bar = bar
        self.bar_len = 30
        self.progressbar = progressbar

    def map(self, func: Callable, core: int = 1) -> 'ChainIter':
        """
        Chainable map.

        Parameters
        ----------
        func: Callable
            Function to run.
        core: int
            Number of cpu cores.
            If it is larger than 1, multiprocessing based on
            multiprocessing.Pool will be run.
            And so, If func cannot be lambda or coroutine if
            it is larger than 1.
        Returns
        ---------
        ChainIter with result

        >>> ChainIter([5, 6]).map(lambda x: x * 2).get()
        [10, 12]
        """
        logger.info(' '.join(('Running', str(func.__name__))))
        if (core == 1):
            return ChainIter(map(func, self.data), False, self.max,
                             self.bar, self.progressbar)
        with Pool(core) as pool:
            result = pool.map_async(func, self.data).get()
        return ChainIter(result, True, self.max, self.bar, self.progressbar)

    def starmap(self, func: Callable, core: int = 1) -> 'ChainIter':
        """
        Chainable starmap.
        In this case, ChainIter.data must be Iterator of iterable objects.

        Parameters
        ----------
        func: Callable
            Function to run.
        core: int
            Number of cpu cores.
            If it is larger than 1, multiprocessing based on
            multiprocessing.Pool will be run.
            And so, If func cannot be lambda or coroutine if
            it is larger than 1.
        Returns
        ---------
        ChainIter with result
        >>> def multest2(x, y): return x * y
        >>> ChainIter([5, 6]).zip([2, 3]).starmap(multest2).get()
        [10, 18]
        """
        logger.info(' '.join(('Running', str(func.__name__))))
        if core == 1:
            return ChainIter(starmap(func, self.data),
                             False, self.max, self.bar, self.progressbar)
        with Pool(core) as pool:
            result = pool.starmap_async(func, self.data).get()
        return ChainIter(result, True, self.max, self.bar, self.progressbar)

    def filter(self, func: Callable) -> 'ChainIter':
        """
        Simple filter function.
        It kills progress bar.

        Parameters
        ----------
        func: Callable
        """
        logger.info(' '.join(('Running', str(func.__name__))))
        return ChainIter(filter(func, self.data), False, 0, self.bar,
                         self.progressbar)

    def async_map(self, func: Callable, chunk: int = 1) -> 'ChainIter':
        """
        Chainable map of coroutine, for example, async def function.

        Parameters
        ----------
        func: Callable
            Function to run.
        core: int
            Number of cpu cores.
            If it is larger than 1, multiprocessing based on
            multiprocessing.Pool will be run.
            And so, If func cannot be lambda or coroutine if
            it is larger than 1.
        Returns
        ---------
        ChainIter with result
        """
        logger.info(' '.join(('Running', str(func.__name__))))
        if chunk == 1:
            return ChainIter(starmap(run_async,
                                     ((func, a) for a in self.data)),
                             False, self.max, self.bar, self.progressbar)
        with Pool(chunk) as pool:
            result = pool.starmap_async(run_async,
                                        ((func, a) for a in self.data)).get()
        return ChainIter(result, True, self.max, self.bar, self.progressbar)

    def has_index(self) -> bool:
        """
        Return whether it is indexable or not.
        """
        return True if self.indexable else hasattr(self.data, '__getitem__')

    def __getitem__(self, num: int) -> Any:
        if self.has_index():
            return cast(list, self.data)[num]
        self.data = tuple(self.data)
        return self.data[num]

    def reduce(self, func: Callable) -> Any:
        """
        Simple reduce function.

        Parameters
        ----------
        func: Callable

        Returns
        ----------
        Result of reduce.
        """
        logger.info(' '.join(('Running', str(func.__name__))))
        return reduce(func, self.data)

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
        return ChainIter(zip(self.data, *args), False, 0, self.bar,
                         self.progressbar)

    def __iter__(self) -> 'ChainIter':
        self.calc()
        self.max = len(cast(list, self.data))
        return self

    def __next__(self) -> Any:
        if self.bar:
            self.start_time = self.current_time = time.time()
            self.prev_time = self.current_time
            self.current_time = time.time()
            epoch_time = self.current_time - self.prev_time
            if self.max != 0:
                bar_num = int((self.num + 1) / self.max * self.bar_len)
                print(self.progressbar.progress.format(
                    percent=int(100 * (self.num + 1) / self.max),
                    bar=self.progressbar.bar * bar_num,
                    arrow=self.progressbar.arrow,
                    space=self.progressbar.space * (self.bar_len - bar_num),
                    div=' ' + str(self.num + 1) + '/' + str(self.max),
                    epoch_time=round(epoch_time, 3),
                    speed=round(1 / epoch_time, 3)
                    ), end='')
            else:
                print(self.progressbar.cycle.format(
                    cycle=self.progressbar.cycle_token[self.num % 4],
                    div=' ' + str(self.num + 1) + '/' + str(self.max),
                    epoch_time=round(epoch_time, 3),
                    speed=round(1 / epoch_time, 3)
                    ), end='')
        self.num += 1
        if self.num == self.max:
            if self.bar:
                print('\nComplete in {sec} sec!'.format(
                    sec=round(time.time() - self.start_time, 3)))
            raise StopIteration
        return self.__getitem__(self.num - 1)

    def __reversed__(self) -> Iterable:
        if hasattr(self.data, '__reversed__'):
            return cast(list, self.data).__reversed__()
        raise IndexError('Not reversible')

    def __setitem__(self, key: Any, item: Any) -> None:
        if hasattr(self.data, '__setitem__'):
            cast(list, self.data)[key] = item
        raise IndexError('Item cannot be set.')

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

    def calc(self) -> 'ChainIter':
        """
        ChainIter.data may be list, map, filter and so on.
        This method translate it to list.
        If you do not run in parallel, it can print progress bar.

        Returns
        ----------
        ChainIter object with result.
        """
        if self.bar:
            res = []
            start_time = current_time = time.time()
            for n, v in enumerate(self.data):
                res.append(v)
                prev_time = current_time
                current_time = time.time()
                epoch_time = current_time - prev_time
                if self.max != 0:
                    bar_num = int((n + 1) / self.max * self.bar_len)
                    percent = int(100 * (n + 1) / self.max)
                    bar = self.progressbar.bar * bar_num
                    print(self.progressbar.progress.format(
                        percent=percent,
                        bar=bar,
                        arrow=self.progressbar.arrow,
                        space=self.progressbar.space * (self.bar_len
                                                        - bar_num),
                        div=' ' + str(n + 1) + '/' + str(self.max),
                        epoch_time=round(epoch_time, 3),
                        speed=round(1 / epoch_time, 3)
                        ), end='')
                else:
                    print(self.progressbar.cycle.format(
                        cycle=self.progressbar.cycle_token[n % 4],
                        div=' ' + str(n + 1) + '/' + str(self.max),
                        epoch_time=round(epoch_time, 3),
                        speed=round(1 / epoch_time, 3)
                        ), end='')
            print('\nComplete in {sec} sec!'.format(
                sec=round(time.time()-start_time, 3)))
            self.data = res
            return self
        self.data = list(self.data)
        return self

    def __len__(self) -> int:
        self.calc()
        return len(cast(list, self.data))

    def __repr__(self) -> str:
        return 'ChainIter[{}]'.format(str(self.data))

    def __str__(self) -> str:
        return 'ChainIter[{}]'.format(str(self.data))

    def print(self) -> 'ChainIter':
        """
        Just print the content.
        """
        print(self.data)
        return self


if __name__ == '__main__':
    testmod()