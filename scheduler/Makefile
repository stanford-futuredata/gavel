rpc_stubs:
	python -m grpc_tools.protoc -Iruntime/protobuf --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/protobuf/iterator_to_scheduler.proto
	python -m grpc_tools.protoc -Iruntime/protobuf --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/protobuf/scheduler_to_worker.proto
	python -m grpc_tools.protoc -Iruntime/protobuf --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/protobuf/worker_to_scheduler.proto
	python -m grpc_tools.protoc -Iruntime/protobuf --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/protobuf/common.proto
	python -m grpc_tools.protoc -Iruntime/protobuf --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/protobuf/enums.proto

clean:
	rm -rf runtime/rpc_stubs/*_pb2.py runtime/rpc_stubs/*_pb2_grpc.py
