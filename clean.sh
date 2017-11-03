#!/bin/bash

echo 'Removing python cache, checkpoint files and results...'

rm -f __pycache__/*.pyc # compiled python code
rm -f __pycache__/*.pyo # optimized python code
rm -f *.ckpt # checkpoint file(s)
rm -f *.pth # result file(s)

echo 'DONE'
