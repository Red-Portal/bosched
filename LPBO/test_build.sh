g++ -shared -g -O3 -march=native -o libLPBO.so -std=c++17 -fPIC -I/usr/include/eigen3 LPBO.cpp -L/usr/local/lib64/ -lopenblas -llapack -ldlib -fopenmp

