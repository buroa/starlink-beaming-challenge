#!/bin/bash

for f in test_cases/*.txt
do
    echo " > --- "
    echo " > Running scenereo $f ..."
    python3 __init__.py $f > $f.out
    echo " > Evaluating scenereo $f.out ..."
    cat $f.out | python3 ./evaluate.py $f
    rm -rf $f.out
    echo " > --- "
done