# choice

A minimal example of the laya C API. It asks one hardcoded choice question ("Which team should handle this
message?") about one hardcoded message and prints the most likely option.

## Build

Build liblaya first, from the project root:

```sh
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build -j --target laya
```

Then build the example, from this directory:

```sh
cmake -B build
cmake --build build
```

If the project root is somewhere else, pass `-DLAYA_DIR=/path/to/laya-cpp`.
