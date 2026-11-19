#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for i in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null
done
echo "Built report/report.pdf"
ls -la report.pdf