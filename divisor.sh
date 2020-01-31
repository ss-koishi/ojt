#!/bin/sh

path="./images/class-3/OK"
imgs=`find $path -maxdepth 1 -type f`

mkdir -p ./images/class-4/OK
mkdir -p ./images/class-4/NG
mkdir -p ./images/class-4/STOP

for img in $imgs;
do
    echo $img
    open -a preview $img
    read id

    if [ $id -eq 1 ]; then
        mv $img "./images/class-4/OK/"
        echo "mv ${img} ./images/class-4/OK/"
    elif [ $id -eq 2 ]; then
        mv $img "./images/class-4/NG/"
        echo "mv ${img} ./images/class-4/NG/"
    elif [ $id -eq 3 ]; then
        mv $img "./images/class-4/STOP/"
        echo "mv ${img} ./images/class-4/STOP/"
    fi
done
