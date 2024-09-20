#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_json_file>"
    exit 1
fi

input_file="$1"

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found!"
    exit 1
fi

# Remove single-line comments (//) and multi-line comments (/* */) from the JSON
cleaned_json=$(sed -e 's|//.*$||g' \
                   -e '/\/\*/,/\*\//d' \
                   -e 's|/\*[^*]*\*+([^/*][^*]*\*+)*/||g' "$input_file")

echo "$cleaned_json"
