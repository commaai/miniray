# miniray
Miniray is a library for distributed compute across a datacenter. Miniray is designed to dispatch tasks of arbitrary python code through redis. Miniray uses python's *concurrent.futures* API. 

### example
```
import miniray

def is_even(n):
  return n % 2 == 0

x = np.arange(100)
results_loop = [is_even(n) for n in x]

with miniray.Executor(job_name='miniray_example_map') as executor:
  results_map = executor.map(is_even, np.arange(100))

with miniray.Executor(job_name='miniray_example_submit') as executor:
  futures = [executor.submit(is_even, n) for n in x]
  results_submit = [future.result() for future in as_completed(futures)]

for a, b, c in zip(results_loop, results_map, results_submit):
  assert a == b == c
```

### want to use?
If you have tasks that you rant to parallelize across multiple machines, miniray might be right for you! Contact harald@comma.ai if miniray is missing something you would like.


