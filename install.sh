#!/bin/bash

if ! [ -x "$(command -v unzip)" ]; then
  echo "'unzip' could not be found. Please install with \"sudo apt install unzip\". " >&2
  exit 1
fi

if ! [ -x "$(command -v curl)" ]; then
  echo "'curl' could not be found. Please install with \"sudo apt install curl\". " >&2
  exit 1
fi

while true; do
    read -p "Do you wish to download datasets? [y/n]: " yn
    case $yn in
      [Yy]* )
          curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1QKYG58WLDdq4ar690H5qCNiSgI1twKie" > /dev/null
          curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1QKYG58WLDdq4ar690H5qCNiSgI1twKie" -o RGB_Dataset.zip
          rm cookie
          rm -rf data/RGB_Dataset
          unzip RGB_Dataset.zip -d data
          rm RGB_Dataset.zip; 
          break;;
      [Nn]* ) break;;
      * ) echo "Please answer yes or no.";;
    esac
done

while true; do
    read -p "Do you wish to download backbone checkpoints? [y/n]: " yn
    case $yn in
      [Yy]* )
          curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1B53u3WApaicCsaLYZIKjNSrmhe8qKeGP" > /dev/null
          curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1B53u3WApaicCsaLYZIKjNSrmhe8qKeGP" -o backbone_ckpt.zip
          rm cookie
          rm -rf data/backbone_ckpt
          unzip backbone_ckpt.zip -d data
          rm backbone_ckpt.zip; 
          break;;
      [Nn]* ) break;;
      * ) echo "Please answer yes or no.";;
    esac
done

while true; do
    read -p "Do you wish to download pretrained model checkpoints? [y/n]: " yn
    case $yn in
      [Yy]* )
          curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0" > /dev/null
          curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0" -o snapshots.zip
          rm cookie
          rm -rf snapshots
          unzip snapshots.zip
          rm snapshots.zip; 
          break;;
      [Nn]* ) break;;
      * ) echo "Please answer yes or no.";;
    esac
done

if ! [ -x "$(command -v conda)" ]; then
  echo "'conda' could not be found. Please install with Anaconda " >&2
  exit 1
fi

while true; do
    read -p "Do you wish to create conda environment? [y/n]: " yn
    case $yn in
      [Yy]* )
          source ~/.zshrc
          conda create -y -n inspyrenet python=3.8
          "${HOME}/anaconda3/envs/inspyrenet/bin/python" -m pip install -r requirements.txt
          break;;
      [Nn]* ) break;;
      * ) echo "Please answer yes or no.";;
    esac
done

clear