
if [ "$1" == "src/handson/circle.py" ] ; then
    class_name=CreateCircle
elif [ "$1" == "src/handson/square_to_circle.py" ] ; then
    class_name=SquareToCircle
else
    echo "Unkonw file: $1"
    exit 1
fi
manim -pql "$1" "$class_name"
