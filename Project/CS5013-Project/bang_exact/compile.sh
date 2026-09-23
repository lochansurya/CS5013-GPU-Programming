#!/usr/bin/env bash

#Copying the code from the 'code' folder

rm src/parANN.cuh

cp src/code/parANN_skeleton.cuh src/parANN.cuh


#Replacing the place holders with actual parameters
sed -i -e "s/ DATABASE_PLACE_HOLDER/ $1/g" *src/parANN.cuh
sed -i -e "s/ L_PLACE_HOLDER/ $2/g" *src/parANN.cuh
sed -i -e "s/ CHUNKS_PLACE_HOLDER/ $3/g" *src/parANN.cuh

#Compile
make clean
make

