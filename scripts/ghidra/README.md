### Run decompilation script
```
./decompile-headless.sh <binary> <function-name> <output-file>
```

### Test

#### Using script
```
./decompile-headless-test.sh <llvm-dir>  // llvm-dir argument is optional
```

#### Using cmake
```
cmake -B /path/to/build -S /path/to/test
cmake --build /path/to/build
ctest  --output-on-failure  --test-dir /path/to/build
```
