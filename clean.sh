#!/bin/bash

echo 'Removing python cache, checkpoint files and results...'

rm -rf __pycache__
rm -f *.ckpt
rm -f *.pth

echo 'DONE'
