/**

These routines read and process the metafile of the Amazon co-purchasing network,
described in the Stanford Large Network Dataset Collection (SNAP collection) 
maintained by Jure Leskovec:

http://snap.stanford.edu/data/amazon-meta.html

From the SNAP website:

"The data was collected by crawling Amazon website and contains product 
metadata and review information about 548,552 different products (Books, 
music CDs, DVDs and VHS video tapes).

For each product the following information is available:

Title
Salesrank
List of similar products (that get co-purchased with the current product)
Detailed product categorization
Product reviews: time, customer, rating, number of votes, number of people 
        that found the review helpful

The data was collected in summer 2006.
"

Each product is listed in the following format:

Id:   15
ASIN: 1559362022
  title: Wake Up and Smell the Coffee
  group: Book
  salesrank: 518927
  similar: 5  1559360968  1559361247  1559360828  1559361018  0743214552
  categories: 3
   |Books[283155]|Subjects[1000]|Literature & Fiction[17]|Drama[2159]|United States[2160]
   |Books[283155]|Subjects[1000]|Arts & Photography[1]|Performing Arts[521000]|Theater[2154]|General[2218]
   |Books[283155]|Subjects[1000]|Literature & Fiction[17]|Authors, A-Z[70021]|( B )[70023]|Bogosian, Eric[70116]
  reviews: total: 8  downloaded: 8  avg rating: 4
    2002-5-13  cutomer: A2IGOA66Y6O8TQ  rating: 5  votes:   3  helpful:   2
    2002-6-17  cutomer: A2OIN4AUH84KNE  rating: 5  votes:   2  helpful:   1
    2003-1-2  cutomer: A2HN382JNT1CIU  rating: 1  votes:   6  helpful:   1
    2003-6-7  cutomer: A2FDJ79LDU4O18  rating: 4  votes:   1  helpful:   1
    2003-6-27  cutomer: A39QMV9ZKRJXO5  rating: 4  votes:   1  helpful:   1
    2004-2-17  cutomer:  AUUVMSTQ1TXDI  rating: 1  votes:   2  helpful:   0
    2004-2-24  cutomer: A2C5K0QTLL9UAT  rating: 5  votes:   2  helpful:   2
    2004-10-13  cutomer:  A5XYF0Z3UH4HB  rating: 5  votes:   1  helpful:   1

where

*) iId: Product id (number 0, ..., 548551)
*) ASIN: Amazon Standard Identification Number
*) title: Name/title of the product
*) group: Product group (Book, DVD, Video or Music)
*) salesrank: Amazon Salesrank
*) similar: ASINs of co-purchased products (people who buy X also buy Y)
*) categories: Location in product category hierarchy to which the product 
      belongs (separated by |, category id in [])
*) reviews: Product review information: time, user id, rating, total number of 
     votes on the review, total number of helpfulness votes (how many people 
     found the review to be helpful)


*/


#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <vector>

using namespace std;
typedef unsigned int Uint;

typedef struct
{
  string date;
  string customer;
  Uint rating;
  Uint votes;
  Uint helpful;
} amazon_review;

class amazon_record
{
 public:
  Uint Id;
  string ASIN;
  string title;
  Uint salesrank;
  vector<string> similar;
  vector<string> category;
  vector<amazon_review> reviews;

  amazon_record(): Id(0), ASIN(), title(), salesrank(0), similar(), reviews() {}
};

void print_amazon_title(std::ostream &s, const amazon_record &R)
{
    s << R.ASIN << " " << R.title << "\n";
}

void print_amazon_title_and_category(std::ostream &s, const amazon_record &R)
{
    s << R.ASIN ;
    if (R.title.size() > 0)
    {
      s << " \"" << R.title << "\" ";
    }
    if (R.category.size() > 0)
    {
      s << "[" << R.category[0] << "]";
    }
    s << "\n";
}

void print_amazon_similar(std::ostream &s, const amazon_record &R)
{
    s << R.ASIN << "  ";
    for (Uint i=0; i< R.similar.size(); i++)
    {
      s << R.similar[i] << " ";
    }
    s << "\n";
}


std::ostream & operator<<(std::ostream &s, const amazon_record &R)
{
    print_amazon_title_and_category(s, R);
    return s;
}

amazon_record process_amazon_record(istream &f)
{
   amazon_record R;

   static const string Id_s = "Id:";
   static const string ASIN_s = "ASIN:";
   static const string title_s = "title:";
   static const string group_s = "group:";
   static const string salesrank_s = "salesrank:";
   static const string similar_s = "similar:";
   static const string categories_s = "categories:";
   static const string reviews_s = "reviews:";

   bool read_Id = false;
   bool read_ASIN = false;
   bool read_title = false;
   bool read_similar = false;
   bool read_category = false;
   

  //cerr << "Inside process_amazon_record(): \n";

   string line;

   /* eat up blank lines, until first non-blank or EOF */

   while (getline(f, line))
   {
      //cerr << "line length  = " << line.length() << ": ["<< line << "] \n";
      if (line.length() > 1)
                break;
      //cerr << "ate blank line.\n";
   }
  
   do 
   {
      string key;

      if (line.length() <= 1)
         break;

      stringstream s(line);
      s >> key;

      if (key == Id_s)
      {
          s >> R.Id;
          read_Id = true;
      }
      else if (key == ASIN_s)
      {
          s >> R.ASIN;
          read_ASIN = true;
      }

      else if (key == title_s)
      {
          getline(s, R.title);
          // trim any leading or whitespace
          size_t start_pos = R.title.find_first_not_of(" \t\n\r");
          size_t end_pos =   R.title.find_last_not_of(" \t\n\r");
          R.title = R.title.substr(start_pos, end_pos);
          read_title = true;
      }
      else if (key == similar_s)
      {
        Uint N = 0;
        s >>  N;


        string item_code;
        for (Uint i=0; i<N; i++)
        {
            s >> item_code;
            R.similar.push_back(item_code);
        }
        read_similar = true;
      }
      else if (key == categories_s)
      {
        Uint N = 0;
        string category_code;

        s >>  N;
        
        //cerr << "\t #category = " << N << ":\n";
        for (Uint i=0; i<N; i++)
        {
            // s >> category_code;
            // cerr << "\t(" << category_code << ")\n";
            // R.category.push_back(category_code);

            getline(f, category_code);
            // trim any leading or whitespace
            size_t start_pos = category_code.find_first_not_of(" \t\n\r");
            size_t end_pos =   category_code.find_last_not_of(" \t\n\r");
            if (start_pos < end_pos)
              category_code = category_code.substr(start_pos, end_pos);
            R.category.push_back(category_code);

        }
        read_category = true;
      }
    }

    while (getline(f, line));
   
   //cerr << "Inisde: read R.\n";
   //cerr << "Inside: " << R;

   return R;
}

std::istream & operator>>(std::istream &s, amazon_record &R)
{
    //cerr << "inside opeator >> : \n";
    R = process_amazon_record(s);
    return s;
}


int main(int argc, char *argv[])
{
  amazon_record R;

  while (cin >> R)
  {
    cout << R;
  }


  return 0;
}


