#!/bin/bash
cd "$(dirname "$0")"
python3 speculative_pool_daily.py "$@"
