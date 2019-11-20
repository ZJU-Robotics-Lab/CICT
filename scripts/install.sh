#!/bin/bash
OPTIONS="cmake git opencv pybind camera quit"
select opt in $OPTIONS; do
	if [ "$opt" = "cmake" ]; then
		sudo apt-get install cmake

	elif [ "$opt" = "git" ]; then
		sudo apt-get install cmake

	elif [ "$opt" = "opencv" ]; then
		sudo apt-get install libopencv-dev python-opencv

	elif [ "$opt" = "pybind" ]; then
		if [ ! -d "3rdpart" ]; then
			mkdir 3rdpart
		fi
		cd 3rdpart
		git clone https://github.com/pybind/pybind11.git
		cd pybind11
		mkdir build
		cd build
		cmake ..
		make -j16
		sudo make install
		cd ../../

	elif [ "$opt" = "camera" ]; then
		if [ ! -d "3rdpart" ]; then
			mkdir 3rdpart
		fi
		cd 3rdpart
		git clone https://github.com/slightech/MYNT-EYE-D-SDK.git
		cd MYNT-EYE-D-SDK
		make init
		make all -j16
		sudo make install
		cd ..

	elif [ "$opt" = "quit" ]; then
		exit

	else
		echo -e "\033[31mBad Option!!! Please enter 1-6\033[0m" 
	fi
done