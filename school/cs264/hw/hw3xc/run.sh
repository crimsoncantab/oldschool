#!/bin/bash
#
### Using bash
#$ -S /bin/bash
#
### Combine output and error messages
#$ -j y
#$ -o hw3.stdouterr.$JOB_ID
#
### Work in the directory where submitted
#$ -cwd
#
## submit to special GPU queue (uses the above 'ortegpu' PE)
#$ -q gpubatch.q
#
### request maximum of 1 hour of compute time
#$ -l h_rt=01:00:00
#

echo " ############### Script Started ############"


#
# Use modules to setup the runtime environment
#
. /etc/profile

# These need to be the same as when the executable was compiled:
module load compilers/gcc/4.3.3                                             
module load mpi/openmpi/1.2.8/gnu                                           
module load packages/cuda/3.2

export EXECUDA=./bin/mpi_histeq 

export INPUT_FILENAME=./data/furtv.bmp
export OUTPUT_FILENAME=$INPUT_FILENAME.out.bmp

#
# Execute the job
#
\time mpirun -np $1 ./$EXECUDA $INPUT_FILENAME $OUTPUT_FILENAME


exit

############################################
