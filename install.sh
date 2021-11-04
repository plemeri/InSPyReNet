#!/bin/bash

if [[ $1 = '-y' ]]
then
    python utils/install_script.py -y
else
    python utils/install_script.py
fi;