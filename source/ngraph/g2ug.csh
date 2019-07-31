#!/bin/csh
#
# make sure graph files are symmetric (and compact)

foreach file ($argv)
  echo processing $file
  set outputfile = $file:r:r.ug
  gzcat $file |  g2ug  > $outputfile
  gzip $outputfile
end


