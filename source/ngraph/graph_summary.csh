#!/bin/csh
#

foreach file ($argv)
  echo $file `gunzip -c $file | graph_summary -u`
end


