#!/bin/csh
#

foreach file ($argv)
  echo processing $file
  gzcat $file |  cluster_node  > $file:r:r.cng
  gzip $file:r:r.cng
end


