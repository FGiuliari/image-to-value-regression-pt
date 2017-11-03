#!/bin/bash

echo 'Removing python cache, checkpoint files and results...'

rm -rf __pycache__
rm *.ckpt
rm *.pth

echo 'DONE'
