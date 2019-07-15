#!/bin/csh
#
# extract largest weakly connected component from a graph
#
# NOTE: must be a simple, integer graph, with no comments or annotations

if ($#argv > 0) then
  set file = $argv[1]
  cat $file | gcomponents -v | \
    awk 'BEGIN {m=0}(m < NF){ m = NF; s=$0 }END {print s}' | \
    gsubgraph_v $file
endif
 

