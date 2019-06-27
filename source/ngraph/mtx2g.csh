#!/bin/csh
#

foreach file ($argv)
  gzcat $file | mtx2g | g2undirected  > $file:r:r.g
  gzip $file:r:r.g
end


