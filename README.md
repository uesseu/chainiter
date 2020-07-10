# ChainIter
## What is this?

It is iterator object package for me. Multiprocessing can be performed easily.  
Ofcourse, you can use it. This script cannot be harmful, but may be anti-pattern.  
It can...

- Use map, filter, and reduce by method chain.
- Calculate fast for a python code.
- Show progress bar.
- Performe parallel computing.
- Run coroutines in parallel.
- Make partial function and run in parallel.

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
pip install git+http://github.com/uesseu/chainiter
```

## Example and basic usage

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

### Laze methods
In ChainIter, lazy methods are

- map(only when chunk size is not set)
- starmap(only when chunk size is not set)
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

It can be used as iterator.

```python
def test(x):
    return x * 2

for n in ChainIter(range(50)).map(test):
  print(n)
```

### Parallel computing
Multi processing can be used easily.  
In this case, use 2 cores.  

```python
def test(x):
    return x * 2

def filt_test(x):
    x > 40

result = ChainIter(range(50)).map(test).calc(2).get()
```

### Optional parallel computing

map and filter method returns ChainIter object with lazy function,  
but optionally, you can set chunk size and run in parallel.  
If you use it, parallel computing will be performed immediately.  
In this case, map will not be lazy function and calc method  
will not be needed.  

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

### Methods

```python
__init__(self, data:Iterable, indexable:bool=False, max_num:int=0,
	 bar:Optional[ProgressBar]=None)  
```

#### Parameters  

| arg       | type                  |                                                                                     |
|-----------|-----------------------|-------------------------------------------------------------------------------------|
| data      | Iterable              | It need not to be indexable.                                                        |
| indexable | bool                  | If data is indexable, indexable should be True.                                     |
| max_num   | int                   | Length of the iterator.                                                             |
| bar       | Optional[ProgressBar] | Show progress bar. It is fancy, but may be slower. It cannot run with multiprocess. |

Bar should be bool or instance of ProgressBar.
If bar is None, no progress bar will be displayed.
If bar is ProgressBar instance, customized progress bar will be displayed.

### arg

```python
arg(self, func:Callable, *args:Any, **kwargs:Any) -> Any
```

Use ChainIter object as argument.  
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
| arg  | type     |                      |
|------|----------|----------------------|
| func | Callable | Function to run.     |
| core | int      | Number of cpu cores. |

If "cores" is larger than 1, multiprocessing based on
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

### filter(self, func:Callable) -> 'ChainIter'
Simple filter function.  
It kills progress bar.  
  
#### Parameters  
func: Callable  

### get(self, kind:type=<class 'list'>) -> Any
Get data as list.

#### Parameters
| arg  | type     |                 |
|------|----------|-----------------|
| kind | Callable | Kind of output. |

If you want to convert to object which is not list,
you can set it. For example, tuple, dqueue, and so on.

### has_index(self) -> bool

#### Return whether it is indexable or not.

### map(self, func:Callable, core:int=1) -> 'ChainIter'
Chainable map.  
  
#### Parameters  
| arg  | type     |                      |
|------|----------|----------------------|
| func | Callable | Function to run.     |
| core | int      | Number of cpu cores. |

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

### pmap(self, *args, **kwargs) -> 'ChainIter'
Chainable map with partial function.  
In this case, ChainIter.data must be Iterator of iterable objects.  
It applies partial, and it cannot apply parallel computation.  
To apply parallelism, use calc() after apply it.  



### pstarmap(self, *args, **kwargs) -> 'ChainIter'
Chainable map with partial function.  
In this case, ChainIter.data must be Iterator of iterable objects.  
It applies partial, and it cannot apply parallel computation.  
To apply parallelism, use calc() after apply it.  


### print(self) -> 'ChainIter'
Just print the content.
Returns self.

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

### starmap(self, func:Callable, core:int=1) -> 'ChainIter'
Chainable starmap.  
In this case, ChainIter.data must be Iterator of iterable objects.  
  
#### Parameters  
| arg  | type     |                      |
|------|----------|----------------------|
| func | Callable | Function to run.     |
| core | int      | Number of cpu cores. |

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
