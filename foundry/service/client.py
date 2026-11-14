"""Explicit JSON-RPC client. Start/attach the service before calling; no local autostart."""
from __future__ import annotations
import json
import socket
import uuid
from foundry.service.protocol import HOST,PORTFILE


def info():
    return json.loads(PORTFILE.read_text(encoding='utf-8'))


def invoke(message, timeout=900.):
    from foundry.service.schema import parse
    parse(message)
    endpoint=info()
    if 'host' in endpoint:
        from foundry.service import remote
        response=remote.request(endpoint,message,timeout)
    else:
        with socket.create_connection((HOST,endpoint['port']),timeout=10.) as sock:
            sock.settimeout(timeout)
            sock.sendall(json.dumps(message,ensure_ascii=False,allow_nan=False).encode()+b'\n')
            response=json.loads(sock.makefile('rb').readline())
    return response


def call(method,params,timeout=900.):
    response=invoke(dict(jsonrpc='2.0',id=uuid.uuid4().hex,method=method,params=params),timeout)
    if 'error' in response:
        raise RuntimeError(response['error']['message'])
    return response['result']


def stop():
    return call('system.stop',{},10.)
