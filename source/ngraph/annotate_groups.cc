/**

   Takes the .groups output file, together with a vertex label index (.t)
   file to produce a list similar to .groups, but with the node labels
   annotated.

   Usage:

   zcat graph.tg | group_ann graph.groups > graph.groups_ann
  
   The format of (.t) is a node-num followed by the name-label, one perline.

   0  label1
   1  label2
   ..


*/


#include <string>
#include <map>
#include <strstream>
#include <iostream>


using namespace std;

typedef map<unsigned int, string> node_array_type;

void read_labels(istream &f, node_array_type &node_array)
{

    string label;
    string url;
    unsigned int node_num =0;

  while (f)
  {
    f >> node_num;
    f >> ws;
    getline(f, label);

    node_array[node_num] = label;
  }  
}


void print_group(istream &f, const node_array_type &node_array)
{
  string line;
  unsigned int i;

  while (f, getline(f, line), line.length() < 1) {};  //move past blank lines
  
  while (line.length() > 0)
  {
    istrstream s(line);
    line >> i;
    cout << node_array[i] << "\n";
    getline(f, line);
  }
  cout << "\n";
}
  


void print_labels(ostream &f, const node_array_type &node_array)
{
    for ( node_array_type::const_iterator p = node_array.begin();
                p != node_array.end(); p++)
    {

      cout << p->first << " " << p->second << "\n";
    }
}


int main(int argc, char *argv[])
{

  node_array_type node_array;
  
  
  read_labels( std::cin, node_array);    

  istream groupfile;
  groupfile.open(argv[1]);

  while (groupfile)
  {
    print_group(group_file), node_aray);
  }
  
  //print_labels(std::cout, node_array);


}


