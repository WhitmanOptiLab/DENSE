#include "ngraph.hpp"

using namespace NGraph;

template <class node_iterator>
Graph create_complete_graph(node_iterator begin, node_iterator end)
{
    Graph A;

    for (node_iterator p = begin; p != end; p++)
      for (node_iterator q = begin; q!=end; q++)
      {
          if (*p != *q )
          {
              A.insert_edge(*p, *q);
          }
      }

   return A;
}

template <class Graph, class node_iterator>
void add_clique(Graph &A, node_iterator begin, node_iterator end)
{

    for (node_iterator p = begin; p != end; p++)
      for (node_iterator q = begin; q!=end; q++)
      {
          if (*p != *q )
          {
              A.insert_edge(*p, *q);
          }
      }

}

template <class Graph, class node_iterator>
void add_undirected_clique(Graph &A, node_iterator begin, node_iterator end)
{

    for (node_iterator p = begin; p != end; p++)
      for (node_iterator q = begin; q!=end; q++)
      {
          if (*p != *q )
          {
              A.insert_undirected_edge(*p, *q);
          }
      }

}


// add in place to a graph 
//
//(NOTE:  A += B is much more efficient than A = A+B.)
//
Graph &operator+=(Graph &A, const Graph &B)
{
       std::vector<Graph::edge> b = B.edge_list();

       for (std::vector<Graph::edge>::const_iterator t = b.begin(); 
                    t < b.end(); t++)
       {
            A.insert_edge(*t);
       };

      return A;
}

#ifndef for_iterator
#define for_iterator(p,C)  for (p=C.begin(); c!=C.end(); (p)++)
#endif


// run over out-going edges of B and add them to A
//
template <class T>
tGraph<T> &operator+=(tGraph<T> &A, const tGraph<T> &B
{
    typedef typename tGraph<T> Graph;
    Graph::iterator p;
    // run over vertciesn of B
    for_each(p,B)
    { 
        const Graph::vertex &v = Graph::Vertex(p);
        Graph::edge_set &S = Graph::OutEdges(p);
        Graph::edge_set::const_iterator e;
        for_iterator(e, S)
        {
            A.InsertEdge(v, *e);
        }
    }
    return A;
}



#endif

/**
    create the union of two graphs.
*/
template <class T>
tGraph<T> operator+(const tGraph<T> &A, const tGraph<T> &B)
{
    
       // start with the largest graph

       tGraph<T>  U(A);
       std::vector<typename tGraph<T>::edge> b = B.edge_list();
       
       for (std::vector<Graph::edge>::const_iterator t = b.begin(); 
                    t < b.end(); t++)
       {
            U.insert_edge(*t);
       };

    
      return U;
       
}

/**
    remove edges
*/
Graph operator-(const Graph &A, const Graph &B)
{
       Graph  U(A);
       std::vector<Graph::edge> b = B.edge_list();
       
       for (std::vector<Graph::edge>::const_iterator 
                    t = b.begin(); t < b.end(); t++)
       {
            if (U.includes_edge(*t)) 
                //U.remove_edge(t->first, t->second);
                U.remove_edge(*t);
       };

    
      return U;
}


#include <cmath>
double correlation(unsigned int N, const double *x, const double *y)
{

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;

    for (unsigned int i=0; i<N; i++, x++, y++)
    {
        sum_x += *x;
        sum_x2 += *x * *x;
        sum_y += *y;
        sum_y2 += *y * *y;
        sum_xy += *x * *y;
    }


#if 0
    std::cout << "Corrleation: \n";
    std::cout << "\t sum x = " << sum_x << "\n";
    std::cout << "\t sum y = " << sum_y << "\n";
    std::cout << "\t sum xy = " << sum_xy << "\n";
    std::cout << "\t sum x2 = " << sum_x2 << "\n";
    std::cout << "\t sum y2 = " << sum_y2 << "\n";
#endif

    return  (N * sum_xy - (sum_x * sum_y)) / 
      std::sqrt( (N*sum_x2 - (sum_x * sum_x)) * (N*sum_y2 - (sum_y * sum_y)));
}

double avg_degree_correlation(const Graph &A)
{
   std::vector<double> x;
   std::vector<double> y;

   for (Graph::const_iterator p=A.begin(); p!=A.end(); p++)
   {
     x.push_back(Graph::degree(p));

     // now find the average degree of neighbors
     double avg = 0.0;
     const Graph::vertex_set &e = Graph::out_neighbors(p);
     for (Graph::vertex_set::const_iterator p = e.begin(); p != e.end(); p++)
     {
       avg += A.degree(*p);
     }
     avg /= e.size();

     y.push_back( avg );
   }
   return correlation(x.size(), &x[0], &y[0]);
}



double max_degree_correlation(const Graph &A)
{
  std::vector<double> x;
  std::vector<double> y;

    // for each node in the graph

    for (Graph::const_iterator p=A.begin(); p!=A.end(); p++)
    {
        double d  = Graph::degree(p); 
        x.push_back(d);
        // std::cout << d << "  ";

        // now find the average degree of neighbors
        double m = 0.0;
        const Graph::vertex_set &e = Graph::out_neighbors(p);
        for (Graph::vertex_set::const_iterator p = e.begin(); p != e.end(); p++)
        {
            double d = A.degree(*p);
            if (d>m)
                m = d;
        }

        y.push_back( m );
    }

    // std::cout << "call correlation() with N = " << x.size() << "\n";
    return correlation(x.size(), &x[0], &y[0]);
}



double max_sqrt_degree_correlation(const Graph &A)
{
    std::vector<double> x;
    std::vector<double> y;


    // for each node in the graph

    for (Graph::const_iterator p=A.begin(); p!=A.end(); p++)
    {
        double d  = Graph::degree(p); 
        x.push_back(d);
        // std::cout << d << "  ";

        // now find the average degree of neighbors
        double m = 0.0;
        const Graph::vertex_set &e = Graph::out_neighbors(p);
        for (Graph::vertex_set::const_iterator p = e.begin(); p != e.end(); p++)
        {
            double d = A.degree(*p);
            if (d>m)
                m = d;
        }

        y.push_back( sqrt(static_cast<double>(m) ));
    }

    // std::cout << "call correlation() with N = " << x.size() << "\n";
    return correlation(x.size(), &x[0], &y[0]);
}


#if 0
Graph read_adj_list()
{

}

void write_adj_list(const Graph &G)
{
  for (Graph::const_iterator p = G.begin(); p!=G.end(); p++)
  {
      const Graph::vertex_set& E = Graph::out_neighbors(p);
      
}
#endif
