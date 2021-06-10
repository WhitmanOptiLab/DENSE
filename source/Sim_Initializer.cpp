#include "Sim_Initializer.hpp"

namespace dense{

void parse_graphml(const char* file, NGraph::Graph* adj_graph){
  ezxml_t graphml = ezxml_parse_file(file);
  ezxml_t _graph = ezxml_child(graphml, "graph");
  ezxml_t _edges = ezxml_child(_graph, "edges");
  ezxml_t _edge = ezxml_child(_edges, "edge");
  
  if(_edge == nullptr){
    _edge = ezxml_child(_graph, "edge");
  }
  
  while ( _edge ){
    const char* vertex1 = ezxml_attr(_edge, "vertex1");
    const char* vertex2 = ezxml_attr(_edge, "vertex2");
    const char* source = ezxml_attr(_edge, "source");
    const char* target = ezxml_attr(_edge, "target");
    
    std::string begin;
    std::string end;
    
    if(vertex1){
      begin = std::string(vertex1);
      end = std::string(vertex2);
    } else {
      begin = std::string(source);
      end = std::string(target);
    }
    
    if(begin[0] == 'n'){
      begin.erase(begin.begin());
      end.erase(end.begin());
    }
    
    adj_graph->insert_edge_noloop(stoi(begin), stoi(end));

    _edge = ezxml_next(_edge);
  }
}

void create_default_graph(NGraph::Graph* a_graph, int cell_total, int tissue_width){
  for (Natural i = 0; i < cell_total; ++i) {
      bool is_former_edge = i % tissue_width == 0;
      bool is_latter_edge = (i + 1) % tissue_width == 0;
      bool is_even = i % 2 == 0;
      Natural la = (is_former_edge || !is_even) ? tissue_width - 1 : -1;
      Natural ra = !(is_latter_edge || is_even) ? tissue_width + 1 :  1;

      Natural top          = (i - tissue_width      + cell_total) % cell_total;
      Natural bottom       = (i + tissue_width                   ) % cell_total;
      Natural bottom_right = (i                  + ra              ) % cell_total;
      Natural top_left     = (i                  + la              ) % cell_total;
      Natural top_right    = (i - tissue_width + ra + cell_total) % cell_total;
      Natural bottom_left  = (i - tissue_width + la + cell_total) % cell_total;

      if (is_former_edge) {
        a_graph->insert_edge(i,abs(top));
        a_graph->insert_edge(i,abs(top_left));
        a_graph->insert_edge(i,abs(top_right));
        a_graph->insert_edge(i,abs(bottom_left));
      } else if (is_latter_edge) {
        a_graph->insert_edge(i,abs(top));
        a_graph->insert_edge(i,abs(top_right));
        a_graph->insert_edge(i,abs(bottom_right));
        a_graph->insert_edge(i,abs(bottom));
      } else {
        a_graph->insert_edge(i,abs(top_right));
        a_graph->insert_edge(i,abs(bottom_right));
        a_graph->insert_edge(i,abs(bottom));
        a_graph->insert_edge(i,abs(top_left));
        a_graph->insert_edge(i,abs(bottom_left));
    }
  }
}
void graph_constructor(Static_Args_Base* param_args, std::string string_file, int cell_total, int tissue_width){
  NGraph::Graph a_graph;
  if(cell_total == 0 && tissue_width == 0){
    std::ifstream open_file(string_file);
    if( open_file ){
      if(string_file.find("graphml") != std::string::npos){
        parse_graphml(string_file.c_str(), &a_graph);
      }
      else{
        NGraph::Graph a = NGraph::Graph(open_file);
        a_graph = std::move(a);
      }
    } else {
      std::cout << style::apply(Color::red) << "Error: Could not find cell graph file " + string_file + " specified by the -f command. Make sure file is spelled correctly and in the correct directory.\n" << style::reset();
      param_args->help = 2;
    }
    if( a_graph.num_vertices() == 0){
      std::cout << style::apply(Color::red) << "Error: Cell graph from " + string_file + " is invalid. Make sure the graph is declared correctly.\n" << style::reset();
      param_args->help = 2;
    }
  } else {
    create_default_graph(&a_graph, cell_total, tissue_width);
  }
  param_args->adj_graph = std::move(a_graph);
}
  
}
