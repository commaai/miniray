import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import grpc
import pytest
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.http import InferenceServerClient
from tritonclient.utils import InferenceServerException

from miniray.lib import triton_helpers


@pytest.fixture
def triton_servers(monkeypatch):
  loading = threading.Event()
  release = threading.Event()
  requests = []

  class HttpHandler(BaseHTTPRequestHandler):
    def do_GET(self):
      self.send_response(200)
      self.end_headers()

    def do_POST(self):
      self.rfile.read(int(self.headers.get('Content-Length', 0)))
      if self.path.endswith('/load'):
        # Triton's HTTP model load blocks the worker serving other requests.
        loading.set()
        release.wait(10)
      self.send_response(200)
      self.end_headers()
      self.wfile.write(b'[]' if self.path.endswith('/index') else b'{}')

    def log_message(self, *_args):
      pass

  class GrpcHandler(service_pb2_grpc.GRPCInferenceServiceServicer):
    def RepositoryModelLoad(self, request, context):  # noqa: N802
      if request.model_name == 'invalid':
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'invalid model config')
      requests.append((request, context.time_remaining()))
      loading.set()
      release.wait(10)
      return service_pb2.RepositoryModelLoadResponse()  # ty: ignore[unresolved-attribute]

  http_server = HTTPServer(('127.0.0.1', 0), HttpHandler)
  http_thread = threading.Thread(target=http_server.serve_forever)
  http_thread.start()
  with ThreadPoolExecutor(max_workers=2) as executor:
    grpc_server = grpc.server(executor)
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(GrpcHandler(), grpc_server)
    grpc_port = grpc_server.add_insecure_port('127.0.0.1:0')
    grpc_server.start()
    monkeypatch.setattr(triton_helpers, 'TRITON_GRPC_SERVER_ADDRESS', f'127.0.0.1:{grpc_port}')
    try:
      yield f'127.0.0.1:{http_server.server_port}', loading, release, requests
    finally:
      release.set()
      grpc_server.stop(0).wait()
      http_server.shutdown()
      http_thread.join()
      http_server.server_close()


def test_model_load_keeps_http_responsive(triton_servers):
  http_address, loading, release, requests = triton_servers
  config = {'backend': 'python', 'parameters': {'compile': {'string_value': 'true'}}}

  def load_model():
    with InferenceServerClient(http_address) as client:
      triton_helpers.load_triton_model(client, 'slow', config)

  with ThreadPoolExecutor(max_workers=1) as executor:
    result = executor.submit(load_model)
    try:
      assert loading.wait(5)
      assert not result.done()
      with urllib.request.urlopen(f'http://{http_address}/v2/health/live', timeout=1) as response:
        assert response.status == 200
      assert not result.done()
    finally:
      release.set()
    result.result(timeout=5)

  request, deadline = requests[0]
  assert request.model_name == 'slow'
  assert json.loads(request.parameters['config'].string_param) == config
  assert 590 < deadline <= 601


def test_model_load_surfaces_server_errors(triton_servers):
  http_address, *_ = triton_servers
  with InferenceServerClient(http_address) as client:
    with pytest.raises(InferenceServerException, match='invalid model config'):
      triton_helpers.load_triton_model(client, 'invalid', {})
