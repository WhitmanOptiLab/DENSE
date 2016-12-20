
using namespace std;

reaction::reaction(RATETYPE rate_in, RATETYPE delay_in, vector<ReactionTerm> inputs_in, vector<ReactionTerm> outputs_in){
    rate=rate_in;
    delay=delay_in;
    inputs=inputs_in;
    outputs=outputs_in;
}

