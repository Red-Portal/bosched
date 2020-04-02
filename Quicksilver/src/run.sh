CURRENT_PATH=/home/msca8h/Projects/bayesian-loop-scheduling
export LD_LIBRARY_PATH=$CURRENT_PATH"/gcc/build/lib64":$CURRENT_PATH
export LIBRARY_PATH=$CURRENT_PATH"/gcc/build/lib64":$CURRENT_PATH
export CXX=$CURRENT_PATH"/gcc/build/bin/g++"
export CC=$CURRENT_PATH"/gcc/build/bin/gcc"

export DEBUG=1
#export OMP_SCHEDULE=BINLPT
#export OMP_SCHEDULE=BO_FSS
export OMP_SCHEDULE=FAC2

ldd qs
