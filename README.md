# Patchestry

[![Build & Test](https://github.com/lifting-bits/patchestry/actions/workflows/ci.yml/badge.svg)](https://github.com/lifting-bits/patchestry/actions/workflows/ci.yml)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the patchestry project.

# Building and installing

See the [BUILDING](BUILDING.md) document.

# Getting Ghidra

Get Java JDK (x64)

```shell
wget -c https://download.oracle.com/java/22/latest/jdk-22_linux-x64_bin.tar.gz -O jdk.tar.gz
tar xvf jdk.tar.gz
mv jdk-22.0.1 ~/jdk
echo "export PATH=\$PATH:~/jdk/bin" >> ~/.bashrc
```

Get Ghidra
```shell
wget -c https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.1.1_build/ghidra_11.1.1_PUBLIC_20240614.zip -O ghidra.zip
unzip ghidra.zip
mv ghidra_11.1.1_PUBLIC ~/ghidra
```

# Installing Ghidra Scripts

Link `ghidra_scripts` directory to `$HOME`. We assume that `./patchestry` contains the cloned repository.
```shell
ln -s patchestry/ghidra_scripts ~
```

# Running patchestry via Ghidra GUI

Make sure patchestry is available via PATH
```shell
patchestry 
```

Start Ghidra GUI
```shell
~/ghidra/ghidraRun
```

Create a project and import a binary file.

Run `PatchestryScript.java`.

# Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) document.

# Licensing

Patchestry is licensed according to the [Apache 2.0](LICENSE) license. Patchestry links against and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with [LLVM exceptions](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT).
