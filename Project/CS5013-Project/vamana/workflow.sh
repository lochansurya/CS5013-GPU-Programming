#!/bin/bash
set -e

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
MAGENTA="\033[0;35m"
CYAN="\033[0;36m"
BOLD="\033[1m"
RESET="\033[0m"

EXEC="./build/fresh_vamana"
ARGS="../data/sift10k/sift10k_randomgraph.bin . ."
REPORT_DIR="report"

usage() {
    echo -e "${CYAN}Usage:${RESET} $0 [--build] [--run] [--ncu] [--nsys]"
    echo -e "  ${YELLOW}--build${RESET}    Configure and compile project"
    echo -e "  ${YELLOW}--run${RESET}      Run executable without profiling"
    echo -e "  ${YELLOW}--ncu${RESET}      Run Nsight Compute profiler"
    echo -e "  ${YELLOW}--nsys${RESET}     Run Nsight Systems profiler"
    echo ""
    echo -e "${CYAN}Examples:${RESET}"
    echo "  $0 --build --run"
    echo "  $0 --run --ncu"
    echo "  $0 --nsys"
    exit 1
}

DO_BUILD=false
DO_RUN=false
DO_NCU=false
DO_NSYS=false

if [[ $# -eq 0 ]]; then
    usage
fi

for arg in "$@"; do
    case "$arg" in
        --build) DO_BUILD=true ;;
        --run)   DO_RUN=true ;;
        --ncu)   DO_NCU=true ;;
        --nsys)  DO_NSYS=true ;;
        -h|--help) usage ;;
        *) echo -e "${RED}[ERROR]${RESET} Unknown option: $arg"; usage ;;
    esac
done

if $DO_BUILD; then
    echo -e "${BLUE}${BOLD}[BUILD]${RESET} Configuring and compiling project..."
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build build
    echo -e "${GREEN}${BOLD}[SUCCESS]${RESET} Build complete."
fi

if [[ ! -f "$EXEC" ]]; then
    echo -e "${RED}${BOLD}[ERROR]${RESET} Executable not found: ${BOLD}$EXEC${RESET}"
    echo -e "        Try running with ${YELLOW}--build${RESET} first."
    exit 1
fi

mkdir -p "$REPORT_DIR"

last_report_num=$(find "$REPORT_DIR" -maxdepth 1 -type f -name "report_*.*" \
    | sed -E 's/.*report_([0-9]+)\..*/\1/' | sort -n | tail -1)

if [[ -z "$last_report_num" ]]; then
    next_report_num=1
else
    next_report_num=$((last_report_num + 1))
fi

if $DO_RUN; then
    echo -e "${MAGENTA}${BOLD}[RUN]${RESET} Running executable..."
    echo -e "${CYAN}${BOLD}$EXEC $ARGS${RESET}"
    "$EXEC" $ARGS
    echo -e "${GREEN}${BOLD}[DONE]${RESET} Execution completed."
fi

if $DO_NCU; then
    if command -v ncu >/dev/null 2>&1; then
        echo -e "${MAGENTA}${BOLD}[NCU]${RESET} Running ${BOLD}Nsight Compute${RESET} profiler..."
        ncu -o "$REPORT_DIR/report_${next_report_num}" "$EXEC" $ARGS
        echo -e "${GREEN}${BOLD}[SAVED]${RESET} Nsight Compute report → ${YELLOW}$REPORT_DIR/report_${next_report_num}.ncu-rep${RESET}"
    else
        echo -e "${RED}${BOLD}[WARN]${RESET} Nsight Compute (ncu) not found."
    fi
fi

if $DO_NSYS; then
    if command -v nsys >/dev/null 2>&1; then
        echo -e "${MAGENTA}${BOLD}[NSYS]${RESET} Running ${BOLD}Nsight Systems${RESET} profiler..."
        nsys profile -o "$REPORT_DIR/report_${next_report_num}" "$EXEC" $ARGS
        echo -e "${GREEN}${BOLD}[SAVED]${RESET} Nsight Systems report → ${YELLOW}$REPORT_DIR/report_${next_report_num}.qdrep${RESET}"
    else
        echo -e "${RED}${BOLD}[WARN]${RESET} Nsight Systems (nsys) not found."
    fi
fi

echo -e "${GREEN}${BOLD}[INFO]${RESET} All requested actions completed."