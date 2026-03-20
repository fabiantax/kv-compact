#!/bin/bash
# safe-llama-run.sh - Wrapper script to prevent stuck llama-cli processes
# Usage: ./safe-llama-run.sh [llama-cli args...]

set -e

LLAMA_CLI="llama-cli.exe"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for existing llama processes
check_stuck_processes() {
    local count=$(ps aux | grep -i 'llama' | grep -v grep | wc -l)

    if [ "$count" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Warning: $count existing llama process(es) found:${NC}"
        ps aux | grep -i 'llama' | grep -v grep
        echo ""
        echo -e "${YELLOW}Attempting to kill...${NC}"

        # Try multiple methods
        taskkill.exe /IM llama-cli.exe /F 2>/dev/null || true
        sleep 2

        # Use PowerShell as fallback
        powershell.exe -Command "Get-Process | Where-Object {$_.ProcessName -like '*llama*'} | Stop-Process -Force" 2>/dev/null || true
        sleep 2

        # Final count
        local final_count=$(ps aux | grep -i 'llama' | grep -v grep | wc -l)
        if [ "$final_count" -gt 0 ]; then
            echo -e "${RED}❌ Failed to kill $final_count process(es). Please use Task Manager.${NC}"
            echo -e "   Press Ctrl+Shift+Esc → Details → End Task on llama-cli.exe"
            return 1
        else
            echo -e "${GREEN}✅ All stuck processes cleaned up.${NC}"
        fi
    fi
    return 0
}

# Find llama-cli binary
find_llama_cli() {
    if [ -f "./build-full/Release/llama-hip-bin/$LLAMA_CLI" ]; then
        echo "./build-full/Release/llama-hip-bin/$LLAMA_CLI"
    elif [ -f "./build/Release/$LLAMA_CLI" ]; then
        echo "./build/Release/$LLAMA_CLI"
    elif [ -f "./$LLAMA_CLI" ]; then
        echo "./$LLAMA_CLI"
    elif command -v $LLAMA_CLI &> /dev/null; then
        echo "$LLAMA_CLI"
    else
        echo -e "${RED}❌ Cannot find $LLAMA_CLI${NC}"
        return 1
    fi
}

# Main execution
main() {
    echo -e "${GREEN}=== Safe llama-cli Runner ===${NC}"
    echo ""

    # Step 1: Check for stuck processes
    if ! check_stuck_processes; then
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo ""

    # Step 2: Find binary
    BINARY=$(find_llama_cli)
    if [ $? -ne 0 ]; then
        exit 1
    fi
    echo -e "${GREEN}Using: $BINARY${NC}"
    echo ""

    # Step 3: Run command
    echo -e "${GREEN}Running: $BINARY $@${NC}"
    echo ""

    # Trap to ensure cleanup on interrupt
    trap 'echo -e "\n${YELLOW}Interrupted. Cleaning up...${NC}"; check_stuck_processes; exit 130' INT

    "$BINARY" "$@"
    EXIT_CODE=$?

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ Command completed successfully${NC}"
    else
        echo -e "${RED}❌ Command exited with code $EXIT_CODE${NC}"
    fi

    # Final cleanup check
    echo ""
    check_stuck_processes

    exit $EXIT_CODE
}

# Run main function
main "$@"
