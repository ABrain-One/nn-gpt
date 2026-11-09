#!/bin/bash
while ! grep -qa "=== DONE in" wlval_run.log 2>/dev/null; do sleep 60; done
echo WLVAL_DONE
