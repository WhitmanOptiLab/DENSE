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

#ifndef _BITVECTOR_H_
#define _BITVECTOR_H_

#include <stdio.h> // for FILE

enum {
  BIT_ERR = -1,
  BIT_OFF = 0,
  BIT_ON = 1,
  NBITS_PER_BYTE = 8
};

typedef unsigned char byte_t;

struct _bitvector {
  byte_t *ptr; // Pointer to a dynamic array
  size_t sz;   // This is the size in bytes of the array pointed to by PTR
};

typedef struct _bitvector bitvector;

/*
  WARNING: If a memory allocation fails when creating
           a new bitvector or when enlarging an
	   existing one, the program in execution
	   is terminated via exit(EXIT_FAILURE) after
	   printing a suitable Out-of-memory message to stderr.
*/

/*
  Return a bitvector whose size in bits is greater 
  or equal than REQUESTEDSIZE or, in case of out
  of memory, a bit vector of size zero
  whose PTR field is the null pointer.

  Remarks: 

  1. The bits of the bitvector are all
     set to zero before the function returns.

  2. If REQUESTEDSIZE is zero, then a bitvector
     with null fields (PTR and SZ) is returned.
*/
bitvector newBitVector (size_t requestedSize);

/*
  If BV is not null, return the size in bits of the bitvector
  pointed to by BV, otherwise return (size_t)(-1).
*/
size_t getBitVectorSize (const bitvector* bv);

/* 
   If BV or BV->PTR is the null pointer, or if POS is greater
   or equal than the size in bits of the bitvector
   pointed to by BV, return BIT_ERR.
   Otherwise, return the value of the bit at position POS
   in the bitvector pointed to by BV, which
   is either BIT_OFF or BIT_ON.
*/
int getBitAtPosition (const bitvector* bv, size_t pos);

/* 
   If BV or BV->PTR is the null pointer, or ENDPOS <= STARTPOS,
   or STARTPOS >= getBitVectorSize(BV), return the null pointer,
   i.e. (int*)0. 
   Otherwise, return the bits from the bitvector pointed to by BV 
   lying in the range  [STARTPOS, ENDPOS) (i.e. bit at STARTPOS included,
   bit at ENDPOS excluded) in the form of a (dynamic) array of integers.

   Remarks:

   1. If the returned pointer is not null, it will always point
      to an array of ENDPOS - STARTPOS elements. In case
      ENDPOS > getBitVectorSize(BV), the last 
      ENDPOS - getBitVectorSize(BV) elements of the array
      will be equal to BIT_ERR.

   2. Once the returned array is not any longer needed,
      the memory allocated for it should be freed to avoid memory leaks.
*/
int* getBitsInRange (const bitvector* bv, size_t startpos, size_t endpos);

/* 
   If BV is not the null pointer, set the bit
   at position POS of the bitvector pointed to by BV
   to the given VALUE. VALUE should be zero for OFF, 
   non-zero for ON.

   Remark:

   If POS is greater or equal than the size in bits
   of the bitvector, then the bitvector is first enlarged
   (through a call to realloc()) to be able to host
   at least POS+1 bits. The newly allocated bits
   are then set to zero, and finally the bit at position POS
   is set to VALUE.
*/
void setBitAtPosition (bitvector* bv, size_t pos, int value);

/* 
   If BV is not the null pointer, among the bits of the bitvector
   pointed to by BV set those ones lying in the range [STARTPOS, ENDPOS) 
   (i.e. bit at STARTPOS included, bit at ENDPOS excluded) 
   to the values specified in the array of integers
   pointed to by VARRAY. 

   Remarks:

   1. No action is performed if STARTPOS >= ENDPOS.

   2. If ENDPOS is larger than the size in bits
      of the bitvector, then the bitvector is first enlarged
      (through a call to realloc()) to be able to host
      at least ENDPOS bits. The newly allocated bits
      are then set to zero, and finally the bits in the range 
      [STARTPOS, ENDPOS) are set accordingly to the values 
      found in the array of integers pointed to by VARRAY. 

   3. VARRAY should point to an array of size >= ENDPOS-STARTPOS,
      otherwise a buffer overrun with possible crash of the calling
      program will occur.
*/
void setBitsInRange (bitvector* bv, size_t startpos, size_t endpos, const int* varray);


/* 
   If BV is not the null pointer, among the bits of the bitvector
   pointed to by BV set those ones lying in the range [STARTPOS, ENDPOS) 
   (i.e. bit at STARTPOS included, bit at ENDPOS excluded) 
   to the given VALUE. VALUE should be zero for OFF, 
   non-zero for ON.

   Remarks:

   1. No action is performed if STARTPOS >= ENDPOS.

   2. If ENDPOS is larger than the size in bits
      of the bitvector, then the bitvector is first enlarged
      (through a call to realloc()) to be able to host
      at least ENDPOS bits. The newly allocated bits
      are then set to zero, and finally the bits in the range 
      [STARTPOS, ENDPOS) are set accordingly to VALUE.
*/
void setBitsInRangeToValue (bitvector* bv, size_t startpos, size_t endpos, int value);


/*
  If MIN is the minimum between ENDPOS and BV->SZ * NBITS_PER_BYTE,
  flip all bits in the range [STARTPOS, MIN)
  (i.e. bit at STARTPOS included, bit at MIN excluded):
  all instances of true (1) become false (0), and all instances of 
  false become true. Return the number of bits actually flipped.

  Remark: if BV == NULL, BV->PTR == NULL, or STARTPOS >= BV->SZ * NBITS_PER_BYTE,
          then no bit is flipped and zero is returned.
*/
size_t flipBitsInRange (bitvector* bv, size_t startpos, size_t endpos);

/* 
   If BV == NULL, BV->PTR == NULL, or BV->SZ is zero, then just return zero.
   Otherwise, print to the file pointed to by FP 
   the contents of the bitvector pointed to by BV.
   In this last case, the returned value shall be the number of bits 
   successfully printed (as '0' or '1') to the file pointed to by FP.
   If this number is less than BV->SZ * NBITS_PER_BYTE, 
   then an I/O error occurred.

   Remark: The contents of the bitvector shall be printed in such a way
           that the leftmost character will represent the highest bit,
           the rightmost character the lowest bit. 
*/
size_t printBitVectorOn (const bitvector* bv, FILE* fp);

/*
  If BV and BV->PTR are not null pointers, then free
  the memory pointed to by BV->PTR and set
  BV->SZ to zero.
*/
void emptyBitVector (bitvector* bv);

#endif /* _BITVECTOR_H_ */
