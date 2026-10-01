#!/bin/bash
# Record a batch of experiments one after another (run inside the container).
#
# Usage:
#   ./run_experiments.sh paper     # the three regimes of the npj robot experiment, 5 min each
#   ./run_experiments.sh sweep     # avoidant 0.1-0.8 and ambivalent 0.1-0.8, 3 min each
#   ./run_experiments.sh all       # paper + sweep
#   ./run_experiments.sh run <avoidant> <ambivalent> <duration_s>
#
# GUI=false hides the Gazebo client and the camera preview window.

GUI="${GUI:-true}"

run_one() {
    echo "=== avoidant=$1 ambivalent=$2 duration=$3s ==="
    roslaunch attachment_model relationship.launch avoidant:="$1" ambivalent:="$2" duration:="$3" gui:="$GUI"
    # let Gazebo and the ROS master exit before the next run
    sleep 5
    pkill -9 gzserver 2>/dev/null
    pkill -9 gzclient 2>/dev/null
    sleep 2
}

paper() {
    run_one 0.0 0.1 300   # secure
    run_one 0.9 0.0 300   # strongly avoidant
    run_one 0.0 0.4 300   # moderately ambivalent
}

sweep() {
    for v in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do run_one "$v" 0.0 180; done
    for v in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do run_one 0.0 "$v" 180; done
}

case "$1" in
    paper) paper ;;
    sweep) sweep ;;
    all)   paper; sweep ;;
    run)   run_one "$2" "$3" "$4" ;;
    *)     grep '^#' "$0" | sed 's/^# \{0,1\}//' | tail -n +2 ;;
esac
