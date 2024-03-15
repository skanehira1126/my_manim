#!/usr/bin/bash

if [ "$1" == "src/handson/circle.py" ] ; then
    class_name=CreateCircle
elif [ "$1" == "src/handson/square_to_circle.py" ] ; then
    class_name=SquareToCircle
elif [ "$1" == "src/handson/animated_square_to_circle.py" ] ; then
    class_name="AnimatedSquareToCircle"
elif [ "$1" == "src/handson/different_rotations.py" ] ; then
    class_name="DifferentRotations"
elif [ "$1" == "src/auc/auc_visualization.py" ] ; then
    class_name="AUCVisualization"
else
    echo "Unkonw file: $1"
    exit 1
fi
manim -pql "$1" "$class_name" 
