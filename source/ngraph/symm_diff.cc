//  Usage:  sym_diff file.a file.b  >  output
//
//  Given a list of words (tokens) find the symmetric set difference
//  between file.a and file.b

#include <iostream>
#include <fstream>
#include <string>
#include "set_ops.hpp"

using namespace std;

int main(int argc, char *argv[])
{
 
  ifstream file_A;
  file_A.open(argv[1]);
  set<string> A;

  while (file_A)
  {
    string line;
    file_A >> line;
    A.insert(line);
  }
  file_A.close();

  ifstream file_B;
  file_B.open(argv[2]);
  set<string> B;

  while (file_B)
  {
    string line;
    file_B >> line;
    B.insert(line);
  }
  file_B.close();

  set<string> D = symm_diff(A, B);


 #if 0 
  for (set<string>::const_iterator p = A.begin(); p != A.end(); p++)
    cout << *p << "\n";
  cout << "\n" ;
    

  for (set<string>::const_iterator p = B.begin(); p != B.end(); p++)
    cout << *p << "\n";
  cout << "\n" ;
#endif

  set<string> DA = A * D;
  for (set<string>::const_iterator p = DA.begin(); p != DA.end(); p++)
    cout << *p << "\n";
  cout << "\n" ;



  set<string> DB = B * D;
  for (set<string>::const_iterator p = DB.begin(); p != DB.end(); p++)
    cout << *p << "\n";
  cout << "\n" ;


  return 0;
}
