#ifndef EQUIVALENCE_H_
#define EQUIVALENCE_H_

/*
  This routine creates equivalence classes for a list of
  pair-wise similar elements.  For example, if we have the
  following relations:

  {a=b, c=d, e=a, f=g, h=d, c=a, h=f }

  This forms the follwing equivalances classes

  {a,b,e,c,d,h} {f,g,h} 


  The input to the routine is a list of pairs, the output is a 
  list (vector) of sets.
*/

#include <iostream>
#include <iterator>
#include <set>
#include <map>

using namespace std;


template <typename T>
void absorb(set<T> &A, set<T> &B)
{

    if (&A == &B)
      return;

    A.insert( B.begin(), B.end());
    B.clear();

    // cerr << "Inside absorb: A = ";
}



template <typename T>
class equivalence
{

    private: 

    map<T, unsigned int> E_;               // map of element and its equiv
                                           // equivalence class number start at
                                           // 1, so if E_[a] == 0 then a is not
                                           // in any equivalance class

    map<unsigned int,  set<T> > S_;        // list of equivalence classes
    unsigned int equivalence_class_num_;   // counter used to create new equiv

  public:

    typedef T value_type;
    typedef set<T> element_set;
    typedef typename map<unsigned int, element_set>::iterator iterator;
    typedef typename map<unsigned int, element_set>::const_iterator 
                                                           const_iterator;

    const_iterator begin() const { return S_.begin(); }
    const_iterator end() const { return S_.end(); }

    equivalence(void): E_(), S_(), equivalence_class_num_(0) {};

    static const element_set &collection(const_iterator p) 
    {
        return p->second;
    }

    static unsigned int index(const_iterator p) 
    {
        return p->first;
    }

    unsigned int num_classes() const
    {
        return S_.size();
    }


    unsigned int index(const T& a) const
    {
      typename map<T, unsigned int>::const_iterator p = E_.find(a);
      return (p== E_.end() ? 0 : p->second);
    }

    const set<T>& operator()(const T& a) const
    {
        static const set<T> empty_set;
        unsigned int i = index(a);
        return (i==0) ? empty_set : S_[i] ;
    }

    unsigned int class_size(unsigned int i) const
    {
      return S_[i].size();
    }


    unsigned int size() const
    {
        return num_elements();
    }

    unsigned int num_elements() const
    {
        return E_.size();

    }
    void insert(const T& a)
    {
        // if a is not in an equivalence class, create a new one
        if (E_[a] ==0)
        {
          unsigned int i = ++equivalence_class_num_;
          E_[a] = i;
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

      if (E_[a] !=0)     // if a is already in an equivalence class,
      {
          if (E_[b] == 0)  // but b isn't
          {          
              E_[b] = E_[a];      // make b's equiv. # the same as a
              S_[E_[a]].insert(b);
          }

          else // both a and b are in E_
          if ( E_[b] != E_[a])  // but different equivalent classes
          {

             //cout << "   E_[a] = " << E_[a] <<", E_[b] = "<<E_[b] << "\n";

             // merge the two equivalence classes together, absorbing
             // b into a:   a <- b 
             //
             set<T> &sa = S_[E_[a]];
             set<T> &sb = S_[E_[b]];
             int indexb = E_[b];

             typename set<T>::const_iterator t = sb.begin(); 
             for (; t !=sb.end(); t++)
             {
                  // cerr << "changing "<< *t << " [" << E_[*t] << "] to " 
                  //      << E_[a] << "\n";
                  E_[*t] = E_[a];

             }
             absorb(sa, sb);
             S_.erase(indexb);
    
          }
        }
        else          // a does not have an equiv. class #, and ...
        {
          if (E_[b] != 0)      // but b has one
          {
              E_[a] = E_[b];
              S_[E_[b]].insert(a);
          }
          else // neither a nor b has one -- so create a enw one! 
          {

              set<T> s;
              s.insert(a);
              s.insert(b);
              unsigned int i = ++equivalence_class_num_;
              //cout<< "creating new class #" << equivalence_class_num << "\n";
              E_[a] = E_[b] = i;   
              S_[i] = s;          // S_[E_[a]] = {a, b}
          }
        }
     }


      // assumes there is at least two elmements in list
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
