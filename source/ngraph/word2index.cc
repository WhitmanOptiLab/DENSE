#include <iostream>
#include <string>
#include <sstream>
#include <map>

/*
    convert a general text file to integers by replacing each word in 
    a line by a unique integer identifier.  For example, the file

    the dog jumped over the fence
    the fence was over the river


    0 1 2 3 0 4 
    0 4 5 3 0 6 


   -t option displays the mapping of words (in alphabetical order) to integers

    1 dog
    4 fence
    2 jumped
    3 over
    6 river
    0 the
    5 was

 Finally, one may sort this by the index, reproducing the order in which
 the words were encountered, e.g.

 cat word.txt | word2index -t | sort -n

 0 the
 1 dog
 2 jumped
 3 over
 4 fence
 5 was
 6 river

*/

using namespace std;

int main(int argc, char *argv[])
{
    bool perform_mapping_only = false;


    if (argc > 1)
      perform_mapping_only = ("-t" == string(argv[1]));
   
    typedef unsigned int UInt;

    typedef map<string, UInt> word_index;
    word_index M;

    std::string line;
    while (getline(cin, line))
    {
          string word;
          stringstream s(line);
          //while (!s.eof())
          while (s >> word)
          {
             word_index::iterator word_loc = M.find(word);
             if (word_loc == M.end())  // word not found
             {
                  // new word; assign it a new number
                  UInt new_index = M.size();
                  M[word] = new_index;
             }
             if (!perform_mapping_only)
             {
                 cout << M[word] << " ";
             }
          }
          if (!perform_mapping_only)
          {
            cout << "\n";
          }
    }

   if (perform_mapping_only)
   {
     for (word_index::const_iterator p = M.begin(); p!= M.end(); p++)
     {
        cout <<  p->second << " " << p->first << "\n";
     }
   }


  return 0;
}

