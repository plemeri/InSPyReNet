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
    read -p "Do you wish to download datasets and backbone checkpoints? [y/n]: " yn
    case $yn in
      [Yy]* )
          curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1KkXffb1DEu1be7NO-RPUy1r2bZqJRuYl" > /dev/null
          curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1KkXffb1DEu1be7NO-RPUy1r2bZqJRuYl" -o data.zip
          rm cookie
          rmdir data
          mkdir data
          unzip data.zip -d data
          rm data.zip; 
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
          rmdir snapshots
          mkdir snapshots
          unzip snapshots.zip -d snapshots
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