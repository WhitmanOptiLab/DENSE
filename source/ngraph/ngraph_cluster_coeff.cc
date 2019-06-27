#ifndef CLUSTER_COEFF_H
#define CLUSTER_COEFF_H

#include "ngraph.hpp"
#include "set_ops.hpp"


using namespace NGraph;

//
// cluster coefficient at a vertex
//

double cluster_coeff(const Graph &A, const  Graph::vertex &a)
{
	const  Graph::vertex_set &neighbors = A.out_neighbors(a); 
	int n = neighbors.size();

	if (n < 2) return 0.0;

	int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( Graph::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(A.out_neighbors(*p),neighbors);
	}
  return  static_cast<double>(num_edges) / (n*(n-1));
}



double cluster_coeff(const Graph &A,  Graph::const_iterator t)
{
	const  Graph::vertex_set &neighbors = Graph::out_neighbors(t);
	int n = neighbors.size();

	if (n < 2) return 0.0;

	int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( Graph::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(A.out_neighbors(*p),neighbors);
	}
  return  static_cast<double>(num_edges) / (n*(n-1));
}



double cluster_coeff_directed(const Graph &A, const  Graph::vertex &a)
{
	const  Graph::vertex_set &neighbors = A.out_neighbors(a) + A.in_neighbors(a); 
	int n = neighbors.size();

	if (n < 2) return 0.0;

	int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( Graph::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(A.out_neighbors(*p),neighbors);
	}
  return  static_cast<double>(num_edges) / (n*(n-1));
}




double cluster_coeff_directed(const Graph &A,  Graph::const_iterator t)
{

	const  Graph::vertex_set &neighbors = Graph::in_neighbors(t) +  Graph::out_neighbors(t);

	int n = neighbors.size();

	if (n < 2) return 0.0;

	int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( Graph::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(A.out_neighbors(*p),neighbors); 
	}
  return  static_cast<double>(num_edges) / (n*(n-1));
}



//
// cluster coefficient of a graph (is the average of each node)
//

double cluster_coeff(const Graph &A)
{

  double sum = 0.0;
  for ( Graph::const_iterator p = A.begin(); p != A.end(); p++)
  {
      sum += cluster_coeff(A,p); 
  }
  return  sum / A.num_vertices();
}

//
// cluster coefficient of an UNDIRECTED GRAPH where only (i,j) is stored where i<j.
//

double cluster_coeff_undirected(const Graph &A, 
		const  Graph::vertex &a)
{
	const  Graph::vertex_set neighbors = 
								A.out_neighbors(a) + A.in_neighbors(a); 
	int n = neighbors.size();

	if (n < 2) return 0.0;

	int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( Graph::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(A.out_neighbors(*p),neighbors);
	}
  return  (2.0 * num_edges) / (n*(n-1));
}


//
// cluster coefficient of an UNDIRECTED GRAPH where only (i,j) is stored where i<j.
//

double cluster_coeff_undirected(const Graph &A, 
										 Graph::const_iterator t)
{
	const  Graph::vertex_set neighbors = 
					Graph::out_neighbors(t) + Graph::in_neighbors(t);
	int n = neighbors.size();

	if (n < 2) return 0.0;

	int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( Graph::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(A.out_neighbors(*p),neighbors);
	}
  return  (2.0 *num_edges) / (n*(n-1));
}

//
// cluster coefficient of an UNDIRECTED graph 
//

double cluster_coeff_undirected(const Graph &A)
{

  double sum = 0.0;
  for ( Graph::const_iterator p = A.begin(); p != A.end(); p++)
  {
      sum += cluster_coeff_undirected(A,p); 
  }
  return  sum / A.num_vertices();
}



/*   Optimzied version of cluster coeff which uses an existing array
*    of precomputed values.
*/


class cluster_coeff_table
{
	 private:
			Graph &G_;
			double running_avg_;
	 		std::map<Graph::vertex,double> coeff_;

		public:
			Graph & graph() { return G_; }
			double   & avg()   { return running_avg_; }
			double & operator[](const  Graph::vertex &a) 
			{ return coeff_[a]; }

		cluster_coeff_table(Graph &G) : G_(G), running_avg_(0.0), coeff_()
		{
			for ( Graph::const_iterator p=G_.begin(); p!=G_.end(); p++)
							coeff_[Graph::node(p)] = cluster_coeff(G_,Graph::node(p));

			running_avg_ = 0.0;

			for ( std::map<Graph::vertex,double>::const_iterator p=coeff_.begin(); 
							p!=coeff_.end(); p++)
			{		 
								running_avg_ +=  p->second;
			}
		  running_avg_ = running_avg_ / coeff_.size();
		}


		double update(const  Graph::vertex &a)
		{
					 std::map<Graph::vertex, double>::iterator p = coeff_.find(a);
					if (p == coeff_.end()) return 0.0;

				  double old_coeff_a = p->second;
					double new_coeff_a = cluster_coeff(G_,a);

					p->second = new_coeff_a;

					running_avg_ +=  (new_coeff_a - old_coeff_a) / coeff_.size();

					return new_coeff_a;
		}

		// returns the new TOTAL cluster coeff after adding new edge 
		//
	  double update(const  Graph::edge &e)
		{
					const  Graph::vertex &a = e.first;
					const  Graph::vertex &b = e.second;

					// the vertices to recompute the cluster coefficient
					// are "a" and every vertex that points to  "a" and "b".
					//
					Graph::vertex_set v = 
								(G_.in_neighbors(a) * G_.in_neighbors(b));
					v.insert(a);
					//v.insert(b);
					for (Graph::vertex_set::const_iterator p = v.begin(); 
												p!= v.end(); p++)
							update(*p);	
				return running_avg_;
		}


		double update_undirected(const  Graph::vertex &a)
		{
					 std::map<Graph::vertex, double>::iterator p = coeff_.find(a);
					if (p == coeff_.end()) return 0.0;

				  double old_coeff_a = p->second;
					double new_coeff_a = cluster_coeff_undirected(G_,a);

					p->second = new_coeff_a;

					running_avg_ +=  (new_coeff_a - old_coeff_a) / coeff_.size();

					return new_coeff_a;
		}
		// returns the new TOTAL cluster coeff after adding new edge 
		//
	  double update_undirected(const  Graph::edge &e)
		{
					const  Graph::vertex &a = e.first;
					const  Graph::vertex &b = e.second;

					// the vertices to recompute the cluster coefficient
					// are "a", "b" and every vertex that has "a" and "b"
					// as neighbors.
					//

					 Graph::vertex_set a_neighbors = 
							(G_.in_neighbors(a) + G_.out_neighbors(a));
					 Graph::vertex_set b_neighbors = 
								(G_.in_neighbors(b) + G_.out_neighbors(b));
					 Graph::vertex_set v = (a_neighbors * b_neighbors);
					v.insert(a);
					v.insert(b);

					for ( Graph::vertex_set::const_iterator p = v.begin(); p!= v.end(); p++)
							update_undirected(*p);	
				return running_avg_;
		}



};




#endif
// CLUSTER_COEFF_H
