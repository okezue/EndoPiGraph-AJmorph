#!/bin/bash
# Download EGM2 images from BioImage Archive

BASE_URL="https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/540/S-BIAD1540/Files"
DATA_DIR="data/S-BIAD1540/images_egm2"
mkdir -p "$DATA_DIR"

# Read manifest and download
tail -n +2 data/S-BIAD1540/manifest_egm2_full.csv | while IFS=',' read -r image_id path rest; do
    output_file="$DATA_DIR/$(basename "$path")"
    if [ ! -f "$output_file" ]; then
        echo "Downloading: $path"
        curl -s -o "$output_file" "$BASE_URL/$path"
    fi
done
echo "Done downloading EGM2 images"
