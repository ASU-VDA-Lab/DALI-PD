#!/bin/bash

ARCHIVE_FOLDER=$1      # e.g., ./cell_density
TARGET_FOLDER=$2       # e.g., ./cell_density_restored

mkdir -p "$TARGET_FOLDER"

for archive in "$ARCHIVE_FOLDER"/*.tar.bz2; do
  tar -xjf "$archive" -C "$TARGET_FOLDER"
done
