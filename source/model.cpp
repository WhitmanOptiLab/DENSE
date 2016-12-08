

#include "reaction.hpp"

using namespace std;

model::model(){
    reactions= new vector<reaction>;
}


void addtoReaction(reaction toAdd){
    this.reactions.add(toAdd);
}

