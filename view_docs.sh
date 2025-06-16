#!/bin/bash
# Simple documentation viewer for WhisperX

echo "==================================="
echo "WhisperX Documentation Viewer"
echo "==================================="
echo ""

# Function to display menu
show_menu() {
    echo "Select documentation to view:"
    echo ""
    echo "QUICK START:"
    echo "  1) Quick Start Guide (5 minutes)"
    echo "  2) Complete Setup Guide"
    echo "  3) Apple Silicon (MLX) Guide"
    echo ""
    echo "TECHNICAL:"
    echo "  4) MLX Integration Architecture"
    echo "  5) Current Limitations & Fixes"
    echo "  6) Test Results"
    echo ""
    echo "GUIDES:"
    echo "  7) Developer/AI Guide"
    echo "  8) Model Conversion Guide"
    echo "  9) Optimization Roadmap"
    echo ""
    echo "OTHER:"
    echo "  10) Documentation Index"
    echo "  11) Executive Summary"
    echo "  12) Run Benchmarks"
    echo ""
    echo "  0) Exit"
    echo ""
}

# Function to view file
view_file() {
    if [ -f "$1" ]; then
        # Use less if available, otherwise cat
        if command -v less &> /dev/null; then
            less "$1"
        else
            cat "$1"
        fi
    else
        echo "Error: File not found: $1"
        echo "Press Enter to continue..."
        read
    fi
}

# Main loop
while true; do
    clear
    show_menu
    read -p "Enter your choice (0-12): " choice
    
    case $choice in
        1) view_file "docs/quickstart/QUICKSTART.md" ;;
        2) view_file "docs/quickstart/WHISPERX_MLX_SETUP.md" ;;
        3) view_file "docs/quickstart/README_MLX.md" ;;
        4) view_file "docs/technical/MLX_INTEGRATION_PLAN.md" ;;
        5) view_file "docs/technical/CURRENT_LIMITATIONS_AND_FIXES.md" ;;
        6) view_file "docs/technical/MLX_TEST_RESULTS.md" ;;
        7) view_file "docs/guides/AI_DEVELOPER_GUIDE.md" ;;
        8) view_file "docs/guides/MLX_MODEL_CONVERSION_GUIDE.md" ;;
        9) view_file "docs/guides/APPLE_SILICON_OPTIMIZATION_ROADMAP.md" ;;
        10) view_file "docs/README.md" ;;
        11) view_file "EXECUTIVE_SUMMARY_APPLE_SILICON.md" ;;
        12) 
            echo "Running benchmarks..."
            python docs/benchmarks/benchmark_m4_max.py
            echo ""
            echo "Press Enter to continue..."
            read
            ;;
        0) 
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Press Enter to continue..."
            read
            ;;
    esac
done