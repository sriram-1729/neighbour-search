# Neighbour search algorithm in CUDA

Finds out the points that lie within a threshold distance from a given point.

## Description

The input dataset is a file containing space separated values, which are the x, y and z coordinates of a large number of points. So, the data is arranged in the following manner:

32.3 231 92.15<br>
5.1 322.22 112.34<br>
688.22 21 67.8<br>

... and so on. This file **should** be named `input.txt`, since the name of the dataset has been hardcoded in the program. 

Once the program is run, it automatically reads `input.txt` and then waits for user input. This is when we are supposed to type in (as space seperated values) the x, y and z coordinates of the point whose neighbours we want to search, and the search distance. The search distance is the radius of the sphere centered on the user-given point, and the points that lie inside the sphere are defined to be the neighbours of the user-given point. The x, y and z coordinates of these points will be stored in a file named `output.txt` while being space separated. 

The input loop can be terminated by entering an invalid search distance, i.e. by entering 0 or a negative number in the search distance field.

## Sample input and output

Sample input 1:

![Sample input 1](/images/input_1.jpg)

Output (the order of appearance of the points need not be the same):

-136.8 -6.0 42.1<br>
-108.2 4.8 -19.8<br>
-84.6 92.6 49.3<br>
96.0 41.7 -49.8<br>
97.5 80.0 4.4<br>
137.7 10.2 -39.9<br>
107.7 4.4 -6.8<br>

Sample input 2:

![Sample input 2](/images/input_2.jpg)

Output:

-108.2 4.8 -19.8<br>
-100.4 -61.4 -107.5<br>

## Additional notes

* `input.txt` should also lie in the same folder as the executed binary.
* `output.txt` is created in the same folder the executed binary is stored in.
* Whenever a new input is given, the old `output.txt` will get overwritten. If an `output.txt` is not present, a new file with the same name will be created.
* A sample `input.txt` has been provided with 1000 points which has been generated using MATLAB.
* `main.cu` uses host memory and device memory and generally runs faster, but takes more space. `main_v2.cu` uses [Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd). This takes less space but runs slower.

