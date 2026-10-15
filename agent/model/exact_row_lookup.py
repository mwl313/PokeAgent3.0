"""Opt-in first-order CUDA lookup for the tested V100/PyTorch runtime only.

Compilation occurs lazily on the first backward, never on a production import.
No atomics or changed sum order. No support/claim for higher-order gradients.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import torch
from torch.autograd.function import once_differentiable

_modules={}
def _check(code):
    if code:raise RuntimeError(f'experimental exact lookup CUDA driver error {code}')

def _kernel(device):
    if device in _modules:return _modules[device][2]
    if torch.__version__.split('+')[0]!='2.14.0' or torch.version.cuda!='12.6' or torch.cuda.get_device_capability(device)!=(7,0):
        raise RuntimeError('exact row lookup is verified only for PyTorch2.14/cu126/V100')
    nvcc=shutil.which('nvcc')
    if nvcc is None:raise RuntimeError('opt-in exact row lookup requires nvcc')
    source=Path(__file__).with_name('cuda')/'exact_lookup.cu'
    compiler=subprocess.check_output([nvcc,'--version'])
    identity=hashlib.sha256(source.read_bytes()+compiler+b'sm70-fmad-false').hexdigest()
    cache=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'pa3_exact_lookup'
    cache.mkdir(parents=True,exist_ok=True);ptx=cache/f'{identity}.ptx'
    if not ptx.exists():
        temporary=cache/f'{identity}.{os.getpid()}.ptx'
        subprocess.run([nvcc,'--ptx','-arch=sm_70','--fmad=false',str(source),'-o',str(temporary)],check=True,capture_output=True)
        os.replace(temporary,ptx)
    driver=ctypes.CDLL('libcuda.so.1');driver.cuModuleLoad.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_char_p]
    driver.cuModuleGetFunction.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_char_p]
    driver.cuLaunchKernel.argtypes=[ctypes.c_void_p,*([ctypes.c_uint]*7),ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
    module=ctypes.c_void_p();function=ctypes.c_void_p()
    with torch.cuda.device(device):
        _check(driver.cuModuleLoad(ctypes.byref(module),str(ptx).encode()))
        _check(driver.cuModuleGetFunction(ctypes.byref(function),module,b'exact_lookup_sum'))
    _modules[device]=(driver,module,(driver,function))
    return driver,function

class _ExactRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx,table,index):
        if table.device.type!='cuda' or table.dtype!=torch.float32 or index.dtype!=torch.long or index.device!=table.device:
            raise ValueError('exact row lookup requires CUDA FP32 table and matching int64 indices')
        ctx.shape=table.shape;ctx.save_for_backward(index)
        return torch.nn.functional.embedding(index,table)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        index,=ctx.saved_tensors
        with torch.cuda.device(grad.device):
            keys,positions=torch.sort(index.reshape(-1),stable=True)
            unique,counts=torch.unique_consecutive(keys,return_counts=True)
            offsets=counts.cumsum(0)-counts
            output=torch.zeros(ctx.shape,device=grad.device,dtype=grad.dtype)
            if unique.numel():
                driver,function=_kernel(grad.device.index);grad=grad.contiguous()
                args=[ctypes.c_void_p(x.data_ptr()) for x in (grad,output,unique,positions,counts,offsets)]+[ctypes.c_longlong(ctx.shape[1])]
                packed=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args])
                _check(driver.cuLaunchKernel(function,len(unique),(ctx.shape[1]+3)//4,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),packed,None))
        return output,None

def exact_rows(table,index):
    return _ExactRows.apply(table,index)

def prepare_exact_lookup(device):
    """Explicit opt-in warmup before collection/timing; never called by default."""
    torch.cuda.current_stream(device)
    _kernel(torch.device(device).index)
