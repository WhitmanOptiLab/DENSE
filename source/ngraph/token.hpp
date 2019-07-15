

#include <string>

using namespace std;

// modifed from Section 7.3 of 
// http://www.oopweb.com/CPP/Documents/CPPHOWTO/
//                        Volume/C++Programming-HOWTO-7.html
//
template <class StringContainer>
void  Tokenize(const string &S, 
							StringContainer &tokens, 
							const string& delimiters = " \t\n")
{
   // Skip delimiters at beginning.
    string::size_type lastPos = S.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos     = S.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(S.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = S.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = S.find_first_of(delimiters, lastPos);
    }
}


/**

  Tokenizer dispenses token from given string.  If there are no more tokens
  in the string, it returns the empty string.  For example,
 

  Tokenizer T("foo bar ! 3");  
  std::string word = "";

  while ( ! (word = T.next()).empty())
  {
      std::cout << word << "\n";

  }

 prints out

 foo
 bar
 !
 3


 If a specific set of delimiters is needed, they can be passed
 in the constructor, e.g. to parse "23:14:18:69" as separate numbers
 one can write

 Tokenizer T("23:14:18:69", ":");

 T.next() yields "23"
 T.next() yields "14"
 ...



*/
/*inline*/  class Tokenizer
{

 public: 
		Tokenizer(const string &S, const string &delimiters=" \t\n") : S_(S),
				delimiters_(delimiters), begin_pos(0), end_pos(0)
		{
      begin_pos = S_.find_first_not_of(delimiters_, 0);
      end_pos     = S_.find_first_of(delimiters_, begin_pos);
			//cerr << "string: [" << S_ << "]\n";
			//cerr << "delimiters : [" << delimiters_ << "]\n";
		}

		string next()
		{
				string next_token;

				if (begin_pos != string::npos  || end_pos != string::npos)	
				{
					 next_token = S_.substr(begin_pos, end_pos - begin_pos);
					 begin_pos = S_.find_first_not_of(delimiters_, end_pos);
					 end_pos = S_.find_first_of(delimiters_, begin_pos);
				}
				return next_token;
		}

	private:

		const string S_;
		const string delimiters_;
		size_t begin_pos;
		size_t end_pos;

};

