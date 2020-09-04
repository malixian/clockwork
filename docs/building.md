# Building Clockwork

Make sure you have completed the [Installation Pre-Requisites](prerequisites.md)

Check out Clockwork

```
git clone https://gitlab.mpi-sws.org/cld/ml/clockwork.git
```

Build Clockwork

```
cd clockwork
mkdir build
cd build
cmake ..
make -j $(nproc)
```

## Troubleshooting


### Protobuf compiler version doesn't match library version

Installing protocol buffers is annoying.  The compiler version must match the library version.  Check where the `protoc` command leads to (`which protoc`).  Applications like `conda` sometimes install their own, different version of the protocol buffers compiler.  If you are on a Google Cloud VM, modify your `PATH` variable to remove conda.

### G++ version

Compilation can fail with versions of g++ less than 8.