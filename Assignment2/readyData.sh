#!/bin/bash

wget http://archives.textfiles.com/stories.zip
unzip -d ./ stories.zip
rm stories.zip
mv stories/index.html index.html
