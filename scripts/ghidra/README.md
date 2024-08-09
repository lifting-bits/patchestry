### Run decompilation script
```
./decompile-headless.sh <binary> <function-name> <output-file>
```

### Test
Testing framework uses `FileCheck` verifier to verify the output file. Please make sure it is installed at the path.

#### Using script
```
./decompile-headless-test.sh
```

#### Using cmake
```
cmake -B /path/to/build -S /path/to/test
cmake --build /path/to/build
ctest  --output-on-failure  --test-dir /path/to/build
```
