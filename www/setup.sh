#!/usr/bin/env bash

# Check if the current directory is a Git repository
if !( [ -d .git ] || git rev-parse --git-dir > /dev/null 2>&1 ); then
    echo "Error: Script must be run from the root of repository."
    exit 1
fi

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <destination> <docs_build_dir>"
    exit 1
fi

dst="$1"
build="$2"

# Check if the build directory exists
if [ ! -d "$build" ]; then
    echo "Error: Build directory '$build' not found."
    exit 1
fi

if [ ! -d "$build/docs" ]; then
    echo "Error: Build docs directory '$build/docs' not found."
    echo "Info: Probably you did not build `mlir-doc` target."
    exit 1
fi

# Check if the destination directory exists, create it if not
if [ ! -d "$dst" ]; then
    echo "Creating destination directory: $dst"
    mkdir -p "$dst"
fi


# Setup hand-written docs
cp -rv $(pwd)/docs $dst
cp -rv $(pwd)/LICENSE $dst/docs
cp -rv $(pwd)/CONTRIBUTING.md $dst/docs

# Setup auto-generated docs
cp -rv $build/docs $dst

# Setup site assets
cp -rv $(pwd)/www/assets $dst
cp -rv $(pwd)/www/GettingStarted $dst/docs
cp -rv $(pwd)/www/img $dst/docs
cp -rv $(pwd)/www/mkdocs.yml $dst

cp -rv $(pwd)/scripts/ghidra/README.md $dst/docs/GettingStarted/ghidra.md