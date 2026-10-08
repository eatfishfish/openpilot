# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
from libc.stdint cimport uintptr_t

from msgq.visionipc.visionipc cimport cl_mem
from msgq.visionipc.visionipc_pyx cimport VisionBuf, CLContext as BaseCLContext
from .commonmodel cimport CL_DEVICE_TYPE_DEFAULT, cl_get_device_id, cl_create_context, cl_release_context
from .commonmodel cimport mat3, ModelFrame as cppModelFrame, DrivingModelFrame as cppDrivingModelFrame, MonitoringModelFrame as cppMonitoringModelFrame, YoloFrame as cppYoloFrame


cdef class CLContext(BaseCLContext):
  def __cinit__(self):
    self.device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT)
    self.context = cl_create_context(self.device_id)

  def __dealloc__(self):
    if self.context:
      cl_release_context(self.context)

cdef class CLMem:
  @staticmethod
  cdef create(void * cmem):
    mem = CLMem()
    mem.mem = <cl_mem*> cmem
    return mem

  @property
  def mem_address(self):
    return <uintptr_t>(self.mem)

def cl_from_visionbuf(VisionBuf buf):
  return CLMem.create(<void*>&buf.buf.buf_cl)


cdef class ModelFrame:
  cdef cppModelFrame * frame
  cdef int buf_size

  def __dealloc__(self):
    del self.frame

  def prepare(self, VisionBuf buf, float[:] projection):
    cdef mat3 cprojection
    memcpy(cprojection.v, &projection[0], 9*sizeof(float))
    cdef cl_mem * data
    data = self.frame.prepare(buf.buf.buf_cl, buf.width, buf.height, buf.stride, buf.uv_offset, cprojection)
    return CLMem.create(data)

  def buffer_from_cl(self, CLMem in_frames):
    cdef unsigned char * data2
    data2 = self.frame.buffer_from_cl(in_frames.mem, self.buf_size)
    return np.asarray(<cnp.uint8_t[:self.buf_size]> data2)


cdef class DrivingModelFrame(ModelFrame):
  cdef cppDrivingModelFrame * _frame

  def __cinit__(self, CLContext context, int temporal_skip):
    self._frame = new cppDrivingModelFrame(context.device_id, context.context, temporal_skip)
    self.frame = <cppModelFrame*>(self._frame)
    self.buf_size = self._frame.buf_size

cdef class MonitoringModelFrame(ModelFrame):
  cdef cppMonitoringModelFrame * _frame

  def __cinit__(self, CLContext context):
    self._frame = new cppMonitoringModelFrame(context.device_id, context.context)
    self.frame = <cppModelFrame*>(self._frame)
    self.buf_size = self._frame.buf_size


cdef class YoloFrame:
  cdef cppYoloFrame * _frame
  cdef int tensor_size

  def __cinit__(self, CLContext context):
    self._frame = new cppYoloFrame(context.device_id, context.context)
    self.tensor_size = self._frame.tensor_size

  def __dealloc__(self):
    del self._frame

  def process(self, VisionBuf buf, float[:] projection, int left, int top, int right, int bottom):
    cdef mat3 cprojection
    memcpy(cprojection.v, &projection[0], 9*sizeof(float))
    cdef float * data = self._frame.process(
      buf.buf.buf_cl, buf.width, buf.height, buf.stride, buf.uv_offset,
      cprojection, left, top, right, bottom)
    # The YoloFrame owns this buffer for its lifetime. ONNX Runtime consumes
    # the view synchronously before the next camera frame overwrites it.
    return np.asarray(<cnp.float32_t[:self.tensor_size]> data)
