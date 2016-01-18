#!/bin/bash

if [ "$#" -lt "1" ]; then
    echo "Usage $0 indir [numreducers]"
    exit
fi

function run_pr_step {
hadoop jar $SJAR $FILE $NUMRED \
    -input $INPUT \
    -output $OUTPUT \
    -mapper "$MAPPER" \
    -reducer "$REDUCER"
}


SJAR='/usr/lib/hadoop/contrib/streaming/hadoop-streaming-0.20.2+737.jar'
REPEAT='10'
DIR=$1

if [ $2 ]; then
    NUMRED='-numReduceTasks '$2
else
    NUMRED=
fi

###STEP 1 ###
MAPPER='lg.py map'
REDUCER='lg.py reduce'
INPUT=$DIR
OUTPUT=$DIR'-pr-iter-0'
FILE='-file common.py -file lg.py'
run_pr_step


###STEP 2 ###
MAPPER='pr.py map'
REDUCER='pr.py reduce'
FILE='-file common.py -file pr.py'

for i in $(seq 1 $REPEAT)
do
    INPUT=$OUTPUT
    OUTPUT=$DIR'-pr-iter-'$i
    run_pr_step

done

###STEP 3 ###
MAPPER='srt.py map'
REDUCER='cat'
FILE='-file common.py -file srt.py'
NUMRED='-numReduceTasks 1'
INPUT=$OUTPUT
OUTPUT=$DIR'-pr'
run_pr_step


