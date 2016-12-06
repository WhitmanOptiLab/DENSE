#ifndef REACTION_HPP
#define REACTION_HPP



using namespace std;

template<class RATETYPE>
class reaction{
 public:
  typedef std::pair<int, int> ReactionTerm;
    
 private:
  RATETYPE rate;
  RATETYPE delay;
  vector<ReactionTerm> inputs;
  vector<ReactionTerm> outputs;
    
public:
    reaction(RATETYPE rate_in, RATETYPE delay_in, vector<ReactionTerm> inputs_in, vector<ReactionTerm> outputs_in);
};


#endif

