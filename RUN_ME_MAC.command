#!/bin/bash
# Flashcat Betting — Mac launcher.
# Double-click this file in Finder. It does everything from scratch.

set -e
cd "$(dirname "$0")"

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
RESET='\033[0m'

printf "${BLUE}\n========================================================\n"
printf "  FLASHCAT BETTING — Mac launcher\n"
printf "========================================================${RESET}\n\n"

# --- Check Python ---------------------------------------------------------
printf "${BLUE}Checking for Python 3.11+...${RESET}\n"

PYTHON=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
      PYTHON="$candidate"
      printf "${GREEN}  Found $candidate (version $version)${RESET}\n"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  printf "${RED}\n  Could not find Python 3.11 or newer.${RESET}\n\n"
  printf "  Install Python from https://www.python.org/downloads/ then run again.\n"
  read -p "Press Enter to exit..."
  exit 1
fi

# --- Virtualenv ----------------------------------------------------------
if [ ! -d ".venv" ]; then
  printf "${BLUE}\nCreating virtual environment (.venv)...${RESET}\n"
  "$PYTHON" -m venv .venv
fi
source .venv/bin/activate

printf "${BLUE}Installing dependencies (first run takes a minute)...${RESET}\n"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# --- Run pipeline --------------------------------------------------------
printf "${BLUE}\nRunning backtest → reweight → build...${RESET}\n"
PYTHONPATH=src python -m flashcat all \
  --start "${FLASHCAT_BACKTEST_START:-2023-09-01}" \
  --end   "${FLASHCAT_BACKTEST_END:-2024-02-15}" \
  --sport "${FLASHCAT_SPORT:-nfl}" \
  --days-ahead "${FLASHCAT_DAYS_AHEAD:-2}"

printf "${GREEN}\n  ✓ Done. Open docs/index.html in your browser.${RESET}\n\n"
read -p "Press Enter to close..."
