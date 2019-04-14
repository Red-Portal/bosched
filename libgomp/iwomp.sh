#!/bin/bash

#SBATCH -t 168:00:00 
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread

export SLURM_CPU_BIND=verbose
export CPU_BIND=verbose
export CFLAGS=-lm
export LD_LIBRARY_PATH=/storage/shared/msc/openmp_dls/pedroopenmp/libgomp/src/libgomp/build/.libs/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/patrick/Desktop/pedroopenmp/libgomp/src/libgomp/build/.libs/
export WEIGHTS="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
export OMP_NUM_THREADS=20
export GOMP_CPU_AFFINITY="0-19"
echo number of threads
echo $OMP_NUM_THREADS
echo weights of weighted factoring
echo $WEIGHTS

echo ---------------------------------------------------------------------

ARRAY=(static dynamic guided trap wfac fac2 rand)
ARRAY2=("0-19" "0-17 16 17" "0-12 12 11 10 9 8 7 6")
ARRAY3=("./a.out 100" "./a.out 300" "./a.out 500" "./OmpSCR_v2.0/bin/c_md.par 8384 50" "./OmpSCR_v2.0/bin/c_md.par 16384 50" "./rodinia_3.1/openmp/lavaMD/run" "./storage/shared/msc/openmp_dls/NAS/NPB3.3-OMP/bin/mg.B.x" "./storage/shared/msc/openmp_dls/NAS/NPB3.3-OMP/bin/mg.C.x" "./storage/shared/msc/openmp_dls/NAS/NPB3.3-OMP/bin/mg.D.x") 
ARRAY4=("1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1" "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.5 0.5 0.5 0.5 1 1 1" "1 1 1 1 1 1 1 1 1 1 1 1 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1 1 1")
counter0=0

while [ $counter0 -le 5 ]
do
	counter=0
	while [ $counter -le 6 ]
	do
		export OMP_SCHEDULE=${ARRAY[$counter]}        	
		counter2=0
		while [ $counter2 -le 2 ]
		do
			export GOMP_CPU_AFFINITY=${ARRAY2[$counter2]}
			export WEIGHTS=${ARRAY4[$counter2]}
        		echo $GOMP_CPU_AFFINITY
			echo $OMP_SCHEDULE
			eval ${ARRAY3[$counter0]}       
			((counter2++))
		done
	((counter++))
	done
((counter0++))
done

echo ---------------------------------------------------------------------

cd ./2012spec
counter=0
while [ $counter -le 6 ]
do
	export OMP_SCHEDULE=${ARRAY[$counter]}        	
	counter2=0
	while [ $counter2 -le 2 ]
	do
		export GOMP_CPU_AFFINITY=${ARRAY2[$counter2]}
		export WEIGHTS=${ARRAY4[$counter2]}
       		echo $GOMP_CPU_AFFINITY
		echo $OMP_SCHEDULE     	
		sh -c ". ./shrc && runspec --size=ref --noreportable 350.md"
		((counter2++))
	done
((counter++))
done

cd ..

echo ---------------------------------------------------------------------














