# ChainIter
## What is this?

This is an iterator object package. Multiprocessing can be performed easily.  
It can...

- Use map, filter, and reduce by method chain like Node.js.
- Calculate fast for a python code.
- Performe parallel computing.
- Show progress bar(Only when without parallel computing).
- Run coroutines in parallel.
  + Ofcource, it can run asyncroniously!
- Make partial function and run in parallel.

But be careful! It may be anti pattern, because of god class!

## Is it fast for python?

It is based not on list but on iterators.  
I tried to minimize orverhead. And so, usually, it is as fast as iterators of python.  
And so, It may be much faster than list in some cases.
It is a package for me, and I do'nt want slow one ;).  

Ofcource, speed of python is slow.  
If you write a simple python code, it cannot be faster than numpy.  
But calcurating by 1 process may be limited by IO, memory, and so on.  
When it shows progress bar, it may be slow.  

## install

```bash
pip install chainiter
```

## Example and basic usage
It can run normal ans async functions.  

### Normal functions
- At first, define a function
- Second, make ChainIter object and input a data
- Make map by map method
- Use calc method if you used lazy method
- Finally, use get method and get the result

```python
from chaniter import ChainIter

def test(x):
    return x * 2

result = ChainIter(range(50)).map(test).calc().get()
```

### Async functions
- At first, define a async function
- Second, make ChainIter object and input a data
- Make map by async_map method
- Use calc method if you used lazy method
- Finally, use get method and get the result

```python
from chaniter import ChainIter
import asyncio

async def test(x):
    asyncio.sleep(10)
    await asyncio.sleep(10)
    return x * 2

result = ChainIter(range(50)).async_map(test).calc().get()
```

And, you can see, it run asyncroniously.  

### Lazy methods
In ChainIter, lazy methods are

- map(only when chunk size is 1 or not set)
- starmap(only when chunk size is 1 or not set)
- pmap
- pstarmap
- filter
- zip
- product


It is chainable.

```python
def test(x):
    return x * 2

def filt_test(x):
    x > 40

result = ChainIter(range(50)).map(test).filter(filt_test).calc().get()
```

It can be used as iterator in for statement.

```python
def test(x):
    return x * 2

for n in ChainIter(range(50)).map(test):
  print(n)
```

### Parallel computing
Multi processing can be used easily.  
Map and filter method returns ChainIter object with lazy function,  
but, if you set chunk size, it can not be lazy.  
If you use it, parallel computing will be performed immediately.  
When parallel computing is performed, calc method is not be needed.  
But even if calc method is not needed, you can call calc method without overhead.  

In this case, use 2 cores.  
'map(test, 2)' lets test run by 2 cores.  

```python
def test(x):
    return x * 2

def filt_test(x):
    x > 40

result = ChainIter(range(50)).map(test, 2).calc().get()
```

Perhaps, there are functions which needs lots of memory.  
It may be useful if you want to change chunk size flexibly.  

```python
def test(x): return x * 2
result = ChainIter(range(50)).map(test, 2).get() # chunk size=2
```

## CLASSES
- ChainIter: The iterator.
- ProgressBar: Options for progress bar.

## ChainIter
Iterator which can used by method chain like Arry of node.js.  
Multi processing and asyncio can run.  

### init

```python
__init__(self, data:Iterable, indexable:bool=False, max_num:int=0,
	 bar:Optional[ProgressBar]=None)  
```

#### Parameters  

| arg       | type                  |                                                                               |
|-----------|-----------------------|-------------------------------------------------------------------------------|
| data      | Iterable              | It need not to be indexable.                                                  |
| indexable | bool                  | If data is indexable, indexable should be True.                               |
| max_num   | int                   | Length of the iterator.                                                       |
| bar       | Optional[ProgressBar] | Show progress bar. Fancy, but little slower. It cannot run with multiprocess. |

Bar should be bool or instance of ProgressBar.  
If bar is None, no progress bar will be displayed.  
If bar is ProgressBar instance, customized progress bar will be displayed.  

### arg

```python
arg(self, func:Callable, *args:Any, **kwargs:Any) -> Any
```

Use ChainIter object as arguments of function.  
It is same as func(*ChainIter, *args, **kwargs)  

#### Parameters  
| arg  | type     |                 |
|------|----------|-----------------|
| func | Callable | Function to run |

#### Returns  
Result of func(*ChainIter, *args, **kwargs)  

```python
ChainIter([5, 6]).arg(sum)  
11  
```

### async_map(self, func:Callable, chunk:int=1) -> 'ChainIter'
Chainable map of coroutine, for example, async def function.  

#### Parameters  
| arg     | type                   |                                  |
|---------|------------------------|----------------------------------|
| func    | Callable               | Function to run.                 |
| chunk   | int = 1                | Number of cpu cores.             |
| timeout | Optional[float] = None | Time to stop parallel computing. |
| logger  | logging.Logger         | Your favorite logger.            |

If "chunk" is larger than 1, multiprocessing based on
multiprocessing.Pool will be run.  
And so, If func cannot be lambda or coroutine if  
it is larger than 1.  

#### Returns  
ChainIter with result  

### calc(self) -> 'ChainIter'
ChainIter.data may be list, map, filter and so on.  
This method translate it to list.  
If you do not run in parallel, it can print progress bar.  

#### Returns  
ChainIter object with result.  

### filter(self, func:Callable, logger: logging.Logger) -> 'ChainIter'
Simple filter function.  
It kills progress bar.  
  
#### Parameters  
func: Callable  

### get(self, kind:type=<class 'list'>) -> Any
Get data as list.

#### Parameters
| arg        | type     |                 |
|------------|----------|-----------------|
| cores kind | Callable | Kind of output. |

If you want to convert to object which is not list,
you can set it. For example, tuple, dqueue, and so on.

### has_index(self) -> bool

#### Return whether it is indexable or not.

### map(self, func:Callable, chunk:int=1) -> 'ChainIter'
Chainable map.  
  
#### Parameters  
| arg     | type                   |                                  |
|---------|------------------------|----------------------------------|
| func    | Callable               | Function to run.                 |
| chunk   | int=1                  | Number of cpu cores.             |
| timeout | Optional[float] = None | Time to stop parallel computing. |
| logger  | logging.Logger         | Your favorite logger.            |

If it is larger than 1, multiprocessing based on  
multiprocessing.Pool will be run.  
And so, If func cannot be lambda or coroutine if  
it is larger than 1.  

Returns  
ChainIter with result  

```python
ChainIter([5, 6]).map(lambda x: x * 2).get()
[10, 12]
```

### pmap(self, chunk: int = 1, timeout:Optional[float] = None, logger: Logger = logger) -> 'ChainIter'
Partial version of ChainIter.map. It does not return ChainIter object.
It returns a function which returns ChainIter.
Chainable starmap with partial function.
At first, it makes partial function, and then, gets argument of ChainIter.

#### Parameters  

| arg     | type           |                                         |
|---------|----------------|-----------------------------------------|
| chunk   | int            | Number of cores for parallel computing. |
| timeout | int            | Time to stop parallel computing.        |
| logger  | logging.Logger | Your favorite logger.                   |

Returns
A function which returns ChainIter with result


```python
def multest(x, y): return x * y
ChainIter([5, 6]).pmap(4)(multest, 2).get()
```

### pstarpmap(self, chunk: int = 1, timeout:Optional[float] = None, logger: Logger = logger) -> 'ChainIter'
Partial version of ChainIter.starmap. It does not return ChainIter object.
It returns a function which returns ChainIter.
Chainable starmap with partial function.
At first, it makes partial function, and then, gets argument of ChainIter.

| arg     | type           |                                         |
|---------|----------------|-----------------------------------------|
| chunk   | int            | Number of cores for parallel computing. |
| timeout | int            | Time to stop parallel computing.        |
| logger  | logging.Logger | Your favorite logger.                   |

Returns
A function which returns ChainIter with result

```python
def multest(x, y, z): return x * y * z
ChainIter(zip([5, 6], [1, 3])).pstarmap(4)(multest, 2).get()
```


### print(self) -> 'ChainIter'
Just print the content. Returns self.  
It is based on print function.  
If you prefer logging tool, use log function.  

### log(self, logger: logging.Logger, level: int = logging.INFO) -> 'ChainIter'
Use logger and write log.

### reduce(self, func:Callable) -> Any
Simple reduce function.  
  
#### Parameters  
| arg  | type     |                    |
|------|----------|--------------------|
| func | Callable | Function to reduce |
  
#### Returns  
Result of reduce.  

### stararg(self, func:Callable, *args:Any, **kwargs:Any) -> Any
Use ChainIter object as argument.  
It is same as func(*tuple(ChainIter), *args, **kwargs)  
  
#### Parameters  
| arg  | type     |                 |
|------|----------|-----------------|
| func | Callable | Function to run |
  
#### Returns  
ChainIter object  

```python
ChainIter([5, 6]).stararg(lambda x, y: x * y)
30
```

### starmap(self, func:Callable, chunk:int=1, timeout: Optional[float], logger: logging.Logger = logger, *args) -> 'ChainIter'
Chainable starmap.  
In this case, ChainIter.data must be Iterator of iterable objects.  
  
#### Parameters  
| arg     | type           |                                         |
|---------|----------------|-----------------------------------------|
| func    | Callable       | Function to run.                        |
| chunk   | int            | Number of cores for parallel computing. |
| timeout | float          | Time to stop parallel computing.        |
| logger  | logging.Logger | Your favorite logger.                   |
| arg     | type           |                                         |

If 'core' is larger than 1, multiprocessing based on  
multiprocessing.Pool will be run.  
And so, If func cannot be lambda or coroutine if  
it is larger than 1.  

#### Returns  
ChainIter with result  

```python
def multest2(x, y): return x * y
ChainIter([5, 6]).zip([2, 3]).starmap(multest2).get()
[10, 18]
```


### async_pmap(self, chunk:int=1, timeout: Optional[float], logger: logging.Logger = logger, *args) -> Callable:
Partial version of ChainIter.async_map.
It does not return ChainIter object.
It returns a function which returns ChainIter.
Chainable starmap with partial function.
At first, it makes partial function,
and then, gets argument of ChainIter.

| arg     | type           |                                         |
|---------|----------------|-----------------------------------------|
| chunk   | int            | Number of cores for parallel computing. |
| timeout | float          | Time to stop parallel computing.        |
| logger  | logging.Logger | Your favorite logger.                   |
| arg     | type           |                                         |

Returns
A function which returns ChainIter with result

```python
async def multest2(x, y, z): return x * y * z
ChainIter([5, 6]).async_pmap(multest2, 2, 3)().get()
```

### async_pstarmap(self, chunk:int=1, timeout: Optional[float], logger: logging.Logger = logger, *args) -> Callable:
Partial version of ChainIter.async_starmap.
It does not return ChainIter object.
It returns a function which returns ChainIter.
At first, it makes partial function,
and then, gets argument of ChainIter.

| arg     | type           |                                         |
|---------|----------------|-----------------------------------------|
| chunk   | int            | Number of cores for parallel computing. |
| timeout | float          | Time to stop parallel computing.        |
| logger  | logging.Logger | Your favorite logger.                   |
| arg     | type           |                                         |

Returns
ChainIter with result

```python
async def multest2(x, y, z): return x * y * z
ChainIter([5, 6]).zip([2, 3]).async_pstarmap()(multest2, 2).get()
```

### zip(self, *args:Iterable) -> 'ChainIter'
Simple chainable zip function.  
It kills progress bar.  
  
Parameters  
| arg   | type |                   |
|-------|------|-------------------|
| *args | Any  | Iterators to zip. |
  
Returns  
Result of func(*ChainIter, *args, **kwargs)  

## ProgressBar
An container for progressbar.
It is just like this

```python
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
```

## future(func:Callable) -> Callable
Let coroutine return future object. It can be used as decorator.  

## run_async(func:Callable, *args:Any, **kwargs:Any) -> Any
Assemble coroutine and run.

For example...

```
from chainiter import future, run_async

@future
async def hoge():
    return 'fuga'
fuga = run_async(hoge())
```

## run_coroutine(col:Coroutine) -> Any
A function to run coroutine.  
It returns result of coroutine.  
