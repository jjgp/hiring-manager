#!/usr/bin/env bash

set -e

echo "Downloading Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
echo "Installing Git LFS..."
apt-get update
apt-get install -yq git-lfs
git lfs install
echo "Pulling latest with Git LFS..."
git lfs pull
echo "Installing Python requirements..."
pip3.9 install -r requirements.txt
