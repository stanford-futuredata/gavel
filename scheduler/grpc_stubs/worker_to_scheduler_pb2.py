# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: worker_to_scheduler.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import common_pb2 as common__pb2
import enums_pb2 as enums__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='worker_to_scheduler.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x19worker_to_scheduler.proto\x1a\x0c\x63ommon.proto\x1a\x0b\x65nums.proto\"\xa0\x01\n\x15RegisterWorkerRequest\x12.\n\x07\x64\x65vices\x18\x01 \x03(\x0b\x32\x1d.RegisterWorkerRequest.Device\x1aW\n\x06\x44\x65vice\x12\x11\n\tdevice_id\x18\x01 \x01(\r\x12 \n\x0b\x64\x65vice_type\x18\x02 \x01(\x0e\x32\x0b.DeviceType\x12\x18\n\x10\x61vailable_memory\x18\x03 \x01(\x02\"P\n\x16RegisterWorkerResponse\x12\x13\n\tworker_id\x18\x01 \x01(\x04H\x00\x12\x17\n\rerror_message\x18\x02 \x01(\tH\x00\x42\x08\n\x06status\"C\n\x0b\x44oneRequest\x12\x11\n\tworker_id\x18\x01 \x01(\x04\x12\x11\n\tdevice_id\x18\x02 \x01(\r\x12\x0e\n\x06job_id\x18\x03 \x01(\x04\x32t\n\x11WorkerToScheduler\x12\x41\n\x0eRegisterWorker\x12\x16.RegisterWorkerRequest\x1a\x17.RegisterWorkerResponse\x12\x1c\n\x04\x44one\x12\x0c.DoneRequest\x1a\x06.Emptyb\x06proto3')
  ,
  dependencies=[common__pb2.DESCRIPTOR,enums__pb2.DESCRIPTOR,])




_REGISTERWORKERREQUEST_DEVICE = _descriptor.Descriptor(
  name='Device',
  full_name='RegisterWorkerRequest.Device',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='device_id', full_name='RegisterWorkerRequest.Device.device_id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_type', full_name='RegisterWorkerRequest.Device.device_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='available_memory', full_name='RegisterWorkerRequest.Device.available_memory', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=130,
  serialized_end=217,
)

_REGISTERWORKERREQUEST = _descriptor.Descriptor(
  name='RegisterWorkerRequest',
  full_name='RegisterWorkerRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='devices', full_name='RegisterWorkerRequest.devices', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_REGISTERWORKERREQUEST_DEVICE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=57,
  serialized_end=217,
)


_REGISTERWORKERRESPONSE = _descriptor.Descriptor(
  name='RegisterWorkerResponse',
  full_name='RegisterWorkerResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='worker_id', full_name='RegisterWorkerResponse.worker_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error_message', full_name='RegisterWorkerResponse.error_message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='status', full_name='RegisterWorkerResponse.status',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=219,
  serialized_end=299,
)


_DONEREQUEST = _descriptor.Descriptor(
  name='DoneRequest',
  full_name='DoneRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='worker_id', full_name='DoneRequest.worker_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_id', full_name='DoneRequest.device_id', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='job_id', full_name='DoneRequest.job_id', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=301,
  serialized_end=368,
)

_REGISTERWORKERREQUEST_DEVICE.fields_by_name['device_type'].enum_type = enums__pb2._DEVICETYPE
_REGISTERWORKERREQUEST_DEVICE.containing_type = _REGISTERWORKERREQUEST
_REGISTERWORKERREQUEST.fields_by_name['devices'].message_type = _REGISTERWORKERREQUEST_DEVICE
_REGISTERWORKERRESPONSE.oneofs_by_name['status'].fields.append(
  _REGISTERWORKERRESPONSE.fields_by_name['worker_id'])
_REGISTERWORKERRESPONSE.fields_by_name['worker_id'].containing_oneof = _REGISTERWORKERRESPONSE.oneofs_by_name['status']
_REGISTERWORKERRESPONSE.oneofs_by_name['status'].fields.append(
  _REGISTERWORKERRESPONSE.fields_by_name['error_message'])
_REGISTERWORKERRESPONSE.fields_by_name['error_message'].containing_oneof = _REGISTERWORKERRESPONSE.oneofs_by_name['status']
DESCRIPTOR.message_types_by_name['RegisterWorkerRequest'] = _REGISTERWORKERREQUEST
DESCRIPTOR.message_types_by_name['RegisterWorkerResponse'] = _REGISTERWORKERRESPONSE
DESCRIPTOR.message_types_by_name['DoneRequest'] = _DONEREQUEST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RegisterWorkerRequest = _reflection.GeneratedProtocolMessageType('RegisterWorkerRequest', (_message.Message,), dict(

  Device = _reflection.GeneratedProtocolMessageType('Device', (_message.Message,), dict(
    DESCRIPTOR = _REGISTERWORKERREQUEST_DEVICE,
    __module__ = 'worker_to_scheduler_pb2'
    # @@protoc_insertion_point(class_scope:RegisterWorkerRequest.Device)
    ))
  ,
  DESCRIPTOR = _REGISTERWORKERREQUEST,
  __module__ = 'worker_to_scheduler_pb2'
  # @@protoc_insertion_point(class_scope:RegisterWorkerRequest)
  ))
_sym_db.RegisterMessage(RegisterWorkerRequest)
_sym_db.RegisterMessage(RegisterWorkerRequest.Device)

RegisterWorkerResponse = _reflection.GeneratedProtocolMessageType('RegisterWorkerResponse', (_message.Message,), dict(
  DESCRIPTOR = _REGISTERWORKERRESPONSE,
  __module__ = 'worker_to_scheduler_pb2'
  # @@protoc_insertion_point(class_scope:RegisterWorkerResponse)
  ))
_sym_db.RegisterMessage(RegisterWorkerResponse)

DoneRequest = _reflection.GeneratedProtocolMessageType('DoneRequest', (_message.Message,), dict(
  DESCRIPTOR = _DONEREQUEST,
  __module__ = 'worker_to_scheduler_pb2'
  # @@protoc_insertion_point(class_scope:DoneRequest)
  ))
_sym_db.RegisterMessage(DoneRequest)



_WORKERTOSCHEDULER = _descriptor.ServiceDescriptor(
  name='WorkerToScheduler',
  full_name='WorkerToScheduler',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=370,
  serialized_end=486,
  methods=[
  _descriptor.MethodDescriptor(
    name='RegisterWorker',
    full_name='WorkerToScheduler.RegisterWorker',
    index=0,
    containing_service=None,
    input_type=_REGISTERWORKERREQUEST,
    output_type=_REGISTERWORKERRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Done',
    full_name='WorkerToScheduler.Done',
    index=1,
    containing_service=None,
    input_type=_DONEREQUEST,
    output_type=common__pb2._EMPTY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_WORKERTOSCHEDULER)

DESCRIPTOR.services_by_name['WorkerToScheduler'] = _WORKERTOSCHEDULER

# @@protoc_insertion_point(module_scope)
