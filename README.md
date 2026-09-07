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

### Triton model loading

Model loads use gRPC so initialization and compilation do not block Triton's HTTP inference workers.
`TRITON_SERVER_ADDRESS` selects the HTTP endpoint (default `127.0.0.1:8000`).
The gRPC endpoint defaults to the same host on port 8001; set `TRITON_GRPC_SERVER_ADDRESS`
when that port is mapped differently. Model-loading RPCs have a ten-minute deadline.

