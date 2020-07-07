/*
    Numdiff - compare putatively similar files, 
    ignoring small numeric differences
    Copyright (C) 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017  Ivano Primi  <ivprimi@libero.it>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bitvector.h"

static
void exitOnOutOfMemory (int lineNo, size_t nBytes)
{
  fprintf (stderr,
	   "*** Out of memory occurred at line %d of file %s\n"
	   "*** while trying to allocate space for %zu bytes,\n"
	   "*** exit now!\n",
	   lineNo, __FILE__, nBytes);
  exit (EXIT_FAILURE);
}

static
size_t arrangeMemoryFor (bitvector* bv, size_t newSize) 
{
  if (!bv || newSize == 0)
    return 0;
  else
    {
      /*
	BV is not the null pointer and
	NEWSIZE is greater than zero.
      */
      if (!(bv->ptr))
	{
	  bv->ptr = (byte_t*) calloc (newSize, sizeof(byte_t));
	  bv->sz = (!bv->ptr ? 0 : newSize);
	}
      else
	{
	  /* BV->PTR is not the null pointer */
	  if (newSize != bv->sz)
	    {
	      byte_t* tmp = (byte_t*) realloc (bv->ptr, newSize * sizeof(byte_t));
	      if (!tmp)
		{
		  return 0;
		}
	      else
		{
		  bv->ptr = tmp;
		  if (newSize > bv->sz)
		    {
		      memset (bv->ptr + bv->sz * sizeof(byte_t), 0, (newSize-bv->sz) * sizeof(byte_t));
		    }
		  bv->sz = newSize;
		}
	    }
	}
      return bv->sz;
    }
}

static
int getBitFrom (byte_t* barray, size_t pos)
{
  return ( (barray[pos / NBITS_PER_BYTE] & (0x1 << pos % NBITS_PER_BYTE)) == 0
	   ? BIT_OFF : BIT_ON );
}

static
void setBit (byte_t* barray, size_t pos, int value)
{
  if (value == BIT_OFF)
    barray[pos / NBITS_PER_BYTE] &= ~(0x1 << pos % NBITS_PER_BYTE);
  else
    barray[pos / NBITS_PER_BYTE] |= (0x1 << pos % NBITS_PER_BYTE);
}

static
void flipBit (byte_t* barray, size_t pos)
{
  barray[pos / NBITS_PER_BYTE] ^= (0x1 << pos % NBITS_PER_BYTE);
}

bitvector newBitVector (size_t requestedSize)
{
  bitvector bv = {(byte_t*)0, 0};
  size_t sizeInBytes = (requestedSize > 0 ? (requestedSize - 1) / NBITS_PER_BYTE + 1 : 0);

  if ( arrangeMemoryFor (&bv, sizeInBytes) < sizeInBytes )
    {
      exitOnOutOfMemory (__LINE__, sizeInBytes * sizeof(byte_t));
    }
  return bv;
}

size_t getBitVectorSize (const bitvector* bv)
{
  return ((bv) ? NBITS_PER_BYTE * bv->sz : (size_t)(-1));
}

int getBitAtPosition (const bitvector* bv, size_t pos)
{
  if (!bv || !bv->ptr || pos >= bv->sz * NBITS_PER_BYTE)
    {
      return BIT_ERR;
    }
  else
    {
      return getBitFrom (bv->ptr, pos);
    }
}

int* getBitsInRange (const bitvector* bv, size_t startpos, size_t endpos)
{
  if (!bv || !bv->ptr)
    {
      return (int*)0;
    }
  else
    {
      const size_t szInBits = bv->sz * NBITS_PER_BYTE;
      
      if (endpos <= startpos || startpos >= szInBits)
	{
	  return (int*)0;
	}
      else
	{
	  /* 0 <= STARTPOS < ENDPOS, and STARTPOS < SZINBITS */
	  const size_t difference = endpos - startpos;
	  size_t idx;
	  int* barray = (int*) malloc (difference*sizeof(int));
	  
	  if (!barray)
	    {
	      exitOnOutOfMemory (__LINE__, difference*sizeof(int));
	    }
	  for (idx = startpos; idx < szInBits; idx++)
	    {
	      barray[idx-startpos] = getBitFrom (bv->ptr, idx);
	    }
	  for (idx -= startpos; idx < difference; barray[idx++] = BIT_ERR);
	  return barray;
	}
    }
}

void setBitAtPosition (bitvector* bv, size_t pos, int value)
{
  if ((bv))
    {
      if (pos >= bv->sz * NBITS_PER_BYTE)
	{
	  if ( arrangeMemoryFor (bv, pos / NBITS_PER_BYTE + 1) == 0 )
	    {
	      exitOnOutOfMemory (__LINE__, (pos / NBITS_PER_BYTE + 1)*sizeof(byte_t));
	    }
	}
      /* Whenever we arrive here POS < BV->SZ * NBITS_PER_BYTE */
      setBit (bv->ptr, pos, value);
    }
}

void setBitsInRange (bitvector* bv, size_t startpos, size_t endpos, const int* varray)
{
  if ((bv))
    {
      if (endpos > startpos && endpos > bv->sz * NBITS_PER_BYTE)
	{
	  if ( arrangeMemoryFor (bv, (endpos-1) / NBITS_PER_BYTE + 1) == 0 )
	    {
	      exitOnOutOfMemory (__LINE__, ((endpos-1) / NBITS_PER_BYTE + 1)*sizeof(byte_t));
	    }
	}
      size_t pos;

      for (pos = startpos; pos < endpos; pos++)
	{
	  setBit (bv->ptr, pos, varray[pos-startpos]);
	}
    }
}

void setBitsInRangeToValue (bitvector* bv, size_t startpos, size_t endpos, int value)
{
  if ((bv))
    {
      size_t effectiveEnd = endpos;

      if (endpos > startpos && endpos > bv->sz * NBITS_PER_BYTE)
	{
	  if ( arrangeMemoryFor (bv, (endpos-1) / NBITS_PER_BYTE + 1) == 0 )
	    {
	      exitOnOutOfMemory (__LINE__, ((endpos-1) / NBITS_PER_BYTE + 1)*sizeof(byte_t));
	    }
	  if (value == 0)
	    effectiveEnd = bv->sz * NBITS_PER_BYTE;
	}
      size_t pos;

      for (pos = startpos; pos < effectiveEnd; pos++)
	{
	  setBit (bv->ptr, pos, value);
	}
    }
}

size_t flipBitsInRange (bitvector* bv, size_t startpos, size_t endpos)
{
  if ((bv) && (bv->ptr) && bv->sz > 0)
    {
      size_t pos, afterlast = bv->sz * NBITS_PER_BYTE;
      size_t lbd = startpos / NBITS_PER_BYTE + 1;
      size_t ubd;

      if (endpos < afterlast)
	{
	  afterlast = endpos;
	}
      ubd = afterlast / NBITS_PER_BYTE;
      
      if (lbd <= ubd)
	{
	  for (pos = lbd; pos < ubd; pos++)
	    {
	      bv->ptr[pos] = ~(bv->ptr[pos]);
	    }
	  lbd *= NBITS_PER_BYTE;
	  for (pos = startpos; pos < lbd; pos++) 
	    {
	      flipBit (bv->ptr, pos);
	    }
	  ubd *= NBITS_PER_BYTE;
	  for (pos = ubd; pos < afterlast; pos++)
	    {
	      flipBit (bv->ptr, pos);
	    }
	}
      else
	{
	  for (pos = startpos; pos < afterlast; pos++) 
	    {
	      flipBit (bv->ptr, pos);
	    }
	}
      return (startpos < afterlast ? afterlast - startpos : 0);
    }
  else
    {
      return 0;
    }
}

size_t printBitVectorOn (const bitvector* bv, FILE* fp)
{
  if ((bv) && (bv->ptr) && bv->sz > 0)
    {
      const size_t szInBits = bv->sz * NBITS_PER_BYTE;
      size_t pos, idx = 0;
      int bitValue, rv = ' ';
      
      do
	{
	  pos = szInBits - (++idx);
	  bitValue = getBitFrom (bv->ptr, pos);
	  rv = putc ((bitValue == BIT_OFF ? '0' : '1'), fp);
	} while (idx < szInBits && rv != EOF);
      return (rv != EOF ? szInBits : idx-1);
    }
  else
    {
      return 0;
    }
}

void emptyBitVector (bitvector* bv)
{
  if ((bv) && (bv->ptr))
    {
      free ((void*)bv->ptr);
      bv->ptr = (byte_t*)0;
      bv->sz = 0;
    }
}
