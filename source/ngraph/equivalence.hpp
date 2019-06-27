#ifndef EQUIVALENCE_H_
#define EQUIVALENCE_H_

/*
  This routine creates equivalence classes for a list of
  pair-wise similar elements.  For example, if we have the
  following relations:

            [ a, b, c=d, e=a, f=g ]
  
  they form four separate equivalence classes:

               {a,e} {b} {c,d} {f,g}

  and if we add the following relations:

                 [ h=d, c=a ] 

  we get three equivalence classes:

               {b}  {a,c,d,e, h} {f,g}

 and if we continue with two more relationships

                   [ a=f, b=g ]

 all elements collapse into a single equivalence class

                {a,b,c,d,e,f,g,h}




  This optimized version takes into account the relative sizes of
  classes when two different equivalence classes are being joined.
  In cases where one class is significantly larger than the other,
  the large one absorbs the small one.  Thus, the algorithm runs in
  min( size(A), size(B) ).

*/

#include <iostream>
#include <iterator>
#include <set>
#include <map>
#include <vector>

using namespace std;

// move elements form set B into set A
//
template <typename T>
void absorb(set<T> &A, set<T> &B)
{

    if (&A == &B)
      return;

    A.insert( B.begin(), B.end());
    B.clear();

}



template <typename T>
class equivalence
{
  public: 
    typedef T value_type;
    typedef unsigned int index_t;
    typedef set<T> element_set;
    typedef typename map<index_t, element_set>::iterator iterator;
    typedef typename map<index_t, element_set>::const_iterator const_iterator;
    
    // this is used to keep an (optional) merge list of classes
    //
    typedef struct
    {
        index_t left;    // two classes merging into one
        index_t right;   // (left,right) -> (to)
        index_t to;
    } triplet;


    private: 

   map<T, index_t> E_;        // map of element and its equiv
                              // equivalence class number start at
                              // 1, so if E_[a] == 0 then a is not
                              // in any equivalance class

    map<index_t,  set<T> > S_;      // list of equivalence classes
    index_t equivalence_class_num_; // counter for new equiv classes

    // optional record of initial equivalence indices and merges
    bool recording_;
    map<T, index_t> E1_;   // first equivalance index on an element
    vector<triplet> M_;    // merges of equivalence classes (indices)
         
    const set<T> empty_set_;

public:


    const_iterator begin() const { return S_.begin(); }
    const_iterator end() const { return S_.end(); }

    equivalence(void): E_(), S_(), equivalence_class_num_(0), 
          recording_(false), E1_(), M_() {};

    static const element_set &collection(const_iterator p) 
    {
        return p->second;
    }

    static index_t index(const_iterator p) 
    {
        return p->first;
    }

    unsigned int num_classes() const
    {
        return S_.size();
    }


    unsigned int class_size(unsigned int i) const
    {
      typename map<index_t, set<T> >::const_iterator  p = S_.find(i);

      if (p == S_.end())
        return 0;
      else
      return p->second.size();
    }


    unsigned int size() const
    {
        return num_elements();
    }

    unsigned int num_elements() const
    {
        return E_.size();

    }

    bool is_recording() const
    {
        return recording_;
    }

    void recording_on()
    { 
          recording_ = true;
    }

    void recording_off()
    {
        recording_ = false;
    }
    
    
    triplet make_triplet(index_t a, index_t b, index_t c)
    {
        triplet t = {a,b,c};
        return t;
    }


    const vector<triplet> & merge_list() const
    {
        return M_;
    }

    const map<T, index_t> & original_class_indices() const
    {
        return E1_;
    }


     unsigned int index(const T& a) const
     {
       typename map<T, unsigned int>::const_iterator p = E_.find(a);
       return (p== E_.end() ? 0 : p->second);
     }


     bool includes(const T& a) const
     {
          return (index(a) != 0) ;
     }


     const set<T>& operator[](unsigned int i) const
     {
         if (i==0)
         {
            return empty_set_;
         }
         else
         {  
           typename map<index_t, set<T> >::const_iterator p = S_.find(i);
           if (p==S_.end())
              return empty_set_;
            else
              return p->second;
         }
     }   


    bool is_equivalent(const T& a, const T& b) const
    {
        unsigned int ia = index(a);
        if (ia == 0)
          return false;

        unsigned int ib  = index(b);
        if (ib == 0)
          return false;
        else
          return (ia == ib);
    }



     const set<T>& operator()(const T& a) const
     {
        return  this->operator[](index(a));
     }


    void insert(const T& a)
    {
        // if a is not in an equivalence class, create a new one
        if (E_.find(a) == E_.end())
        {
          unsigned int i = ++equivalence_class_num_;
          E_[a] = i;
          if (recording_)
              E1_[a] = i;
          set<T> s;
          s.insert(a);
          S_[i] = s;
        }
    }


    vector<unsigned int> class_sizes() const
    {
        vector<unsigned int> V(num_classes());

        for (const_iterator p = S_.begin(); p!=S_.end(); p++)
        {
            V.push_back( p->second.size());    
        }
        return V;

    }
    void insert(const T& a, const T& b)
    {
        
      // four cases, wether or not a and b are already in equiv classes

      unsigned int E_a = E_[a];
      unsigned int E_b = E_[b];

      if (E_a !=0)     // if a is already in an equivalence class,
      {
          if (E_b == 0)  // but b isn't
          {          
              E_[b] = E_a;      // make b's equiv. # the same as a
              if (recording_)
                  E1_[b] = E_a;
              
              S_[E_a].insert(b);
          }

          else // both a and b are in E_
          if ( E_b != E_a)  // but different equivalent classes
          {

             //cout << "   E_[a] = " << E_[a] <<", E_[b] = "<<E_[b] << "\n";

             // merge the two equivalence classes together, absorbing
             // b into a:   a <- b 
             //

             set<T> &sa = S_[E_a];
             set<T> &sb = S_[E_b];

             unsigned int bigger_class_index = 
                (sa.size() > sb.size() ? E_a : E_b);
             unsigned int smaller_class_index  = 
                  (sa.size() > sb.size() ? E_b : E_a);

             set<T> &bigger_class = (sa.size() > sb.size() ? sa : sb);
             set<T> &smaller_class = (sa.size() > sb.size() ? sb : sa);

             typename set<T>::const_iterator t = smaller_class.begin(); 
             for (; t !=smaller_class.end(); t++)
             {
                  // cerr << "changing "<< *t << " [" << E_[*t] << "] to " 
                  //      << E_[a] << "\n";
                  E_[*t] = bigger_class_index;

             }
             absorb(bigger_class, smaller_class);
             S_.erase(smaller_class_index);
   
             if (recording_)
              M_.push_back( make_triplet(smaller_class_index, 
                        bigger_class_index, bigger_class_index));
          }
          else  // a and b are already in the same equivalence class
          {
             // do nothing 
          }
        }
        else          // a does not have an equiv. class #, and ...
        {
          if (E_[b] != 0)      // but b has one
          {
              E_[a] = E_[b];
              if (recording_)
                  E1_[a] = E_[b];
              S_[E_[b]].insert(a);
          }
          else // neither a nor b has one -- so create a new one! 
          {

              set<T> s;
              s.insert(a);
              s.insert(b);
              unsigned int i = ++equivalence_class_num_;
              //cout<< "creating new class #" << equivalence_class_num << "\n";
              E_[a] = E_[b] = i;   
              if (recording_)
              {
                  E1_[a] = E1_[b] = i;
              }
              S_[i] = s;          // S_[E_[a]] = {a, b}
          }
        }
     }


      // this inserts a list of T's
      //
      // assumes there is at least two elmements in list
      // 
     template<typename const_T_iter>
     void insert( const_T_iter pbegin, const_T_iter pend)
     {
        const_T_iter p = pbegin;
        p++;
        for (; p!=pend; p++)
        {
            insert(*pbegin, *p);
        }
     }
};

template <typename T>
std::ostream & operator<<(std::ostream &s, const equivalence<T> &S)
{
    typedef typename equivalence<T>::element_set  equiv_elm_set;
    typedef typename equivalence<T>::const_iterator equiv_const_iterator;
    typedef typename equiv_elm_set::const_iterator equiv_elm_set_const_iterator;

    for (equiv_const_iterator p = S.begin(); p!=S.end(); p++)
    {
          const equiv_elm_set &E = equivalence<T>::collection(p);
          for (equiv_elm_set_const_iterator e = E.begin(); e!=E.end(); e++)
          {
            s << *e << " " ;
          }
          s << "\n";
    }
    return s;
}

#endif
// EQUIVALENCE_H_
