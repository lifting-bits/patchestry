#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
docker build -t firmware-builder ${script_dir}
docker create --name firmware-builder firmware-builder
docker cp firmware-builder:/output/pulseox-firmware.elf pulseox-firmware.elf
docker cp firmware-builder:/output/bloodlight-firmware.elf bloodlight-firmware.elf
docker rm firmware-builder
