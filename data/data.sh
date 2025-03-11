#!/bin/bash

# File ID from Google Drive
FILE_ID="1Y-CL3C2AEnveI8VJHIzoko0LJOSfR9-L"
OUTPUT_FILE="dataset_time.pkl"

# Download using gdown
gdown --id $FILE_ID -O $OUTPUT_FILE

echo "Download complete: $OUTPUT_FILE"
