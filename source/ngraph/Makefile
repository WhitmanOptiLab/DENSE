.SUFFIXES: .c .cc .o


.c.o:
	$(CC) $(CFLAGS) -c $<

.cc.o:
	$(CCC) $(CFLAGS) -c $<


EXECS = gprune g2degree gcomponents gcluster_coeff mtx2g v2t \
        gremove_v  g2udot g2dot mtx2g gsubgraph_v g2v g2tgf \
        gpagerank k2t gcompact gunion gminus gintersect  \
        mi0_2mi g2ug g2scc gtranspose ifreq g2weak g2degfreq \
        g2Pmetric g2Pmetric_level g2Pmatrix  gecho g2Ptree freq sg2g0 \
        gbigcomponent gneighbors_v t2v g2rand g2prand \
        g2lrand grepeated_edges gg_erdos word2index \
        gcoarsen g2alignment gsize vmap vimap g2giant_v \
        process_amazon

OUTPUT_DIR = $(HOME)/bin

all: $(EXECS) 




CC = icc
CCC = icc
OPTFLAGS = -O3 
WFLAGS =  

LDFLAGS = 
#IFLAGS =   -I$(NGRAPH)

CC = gcc
CCC = g++
OPTFLAGS = -O3 -fomit-frame-pointer -funroll-loops -fforce-mem -fforce-addr -fexpensive-optimizations
OPTFLAGS = -O3 
OPTFLAGS = -O6 -funroll-all-loops
OPTFLAGS = -O6 -funroll-loops
WFLAGS =  -Wall -pedantic  -ansi

DFLAGS = -g
DFLAGS = 

CFLAGS =  $(WFLAGS) $(IFLAGS) $(OPTFLAGS) $(DLFAGS)
OBJS =  

process_amazon : process_amazon.cc
	$(CCC) $(CFLAGS) -o process_amazon process_amazon.cc $(OBJS) $(LDFLAGS) 
	mv process_amazon $(OUTPUT_DIR)

g2giant_v : g2giant_v.cc
	$(CCC) $(CFLAGS) -o g2giant_v g2giant_v.cc $(OBJS) $(LDFLAGS) 
	mv g2giant_v $(OUTPUT_DIR)

g2alignment : g2alignment.cc
	$(CCC) $(CFLAGS) -o g2alignment g2alignment.cc  $(OBJS) $(LDFLAGS) 
	mv g2alignment $(OUTPUT_DIR)

gsize : gsize.cc
	$(CCC) $(CFLAGS) -o gsize gsize.cc  $(OBJS) $(LDFLAGS) 
	mv gsize $(OUTPUT_DIR)


vimap : vimap.cc
	$(CCC) $(CFLAGS) -o vimap vimap.cc  $(OBJS) $(LDFLAGS) 
	mv vimap $(OUTPUT_DIR)


vmap : vmap.cc
	$(CCC) $(CFLAGS) -o vmap vmap.cc  $(OBJS) $(LDFLAGS) 
	mv vmap $(OUTPUT_DIR)



gcoarsen : gcoarsen.cc
	$(CCC) $(CFLAGS) -o gcoarsen gcoarsen.cc  $(OBJS) $(LDFLAGS) 
	mv gcoarsen $(OUTPUT_DIR)

word2index : word2index.cc
	$(CCC) $(CFLAGS) -o word2index word2index.cc  $(OBJS) $(LDFLAGS) 
	mv word2index $(OUTPUT_DIR)

gg_erdos : gg_erdos.cc
	$(CCC) $(CFLAGS) -o gg_erdos gg_erdos.cc  $(OBJS) $(LDFLAGS) 
	mv gg_erdos $(OUTPUT_DIR)

grepeated_edges : grepeated_edges.cc
	$(CCC) $(CFLAGS) -o grepeated_edges grepeated_edges.cc  $(OBJS) $(LDFLAGS) 
	mv grepeated_edges $(OUTPUT_DIR)

g2lrand : g2lrand.cc
	$(CCC) $(CFLAGS) -o g2lrand g2lrand.cc  $(OBJS) $(LDFLAGS) 
	mv g2lrand $(OUTPUT_DIR)

mg2degree : mg2degree.cc
	$(CCC) $(CFLAGS) -o mg2degree mg2degree.cc  $(OBJS) $(LDFLAGS) 
	mv mg2degree $(OUTPUT_DIR)

g2prand : g2prand.cc
	$(CCC) $(CFLAGS) -o g2prand g2prand.cc  $(OBJS) $(LDFLAGS) 
	mv g2prand $(OUTPUT_DIR)

g2rand : g2rand.cc
	$(CCC) $(CFLAGS) -o g2rand g2rand.cc  $(OBJS) $(LDFLAGS) 
	mv g2rand $(OUTPUT_DIR)

gneighbors_v : gneighbors_v.cc
	$(CCC) $(CFLAGS) -o gneighbors_v gneighbors_v.cc  $(OBJS) $(LDFLAGS) 
	mv gneighbors_v $(OUTPUT_DIR)

gbigcomponent : gbigcomponent.cc
	$(CCC) $(CFLAGS) -o gbigcomponent gbigcomponent.cc  $(OBJS) $(LDFLAGS) 
	mv gbigcomponent $(OUTPUT_DIR)


t2v : t2v.cc
	$(CCC) $(CFLAGS) -o t2v t2v.cc  $(OBJS) $(LDFLAGS) 
	mv t2v $(OUTPUT_DIR)

freq : freq.cc
	$(CCC) $(CFLAGS) -o freq freq.cc  $(OBJS) $(LDFLAGS) 
	mv freq $(OUTPUT_DIR)

sg2g0 : sg2g0.cc
	$(CCC) $(CFLAGS) -o sg2g0 sg2g0.cc  $(OBJS) $(LDFLAGS) 
	mv sg2g0 $(OUTPUT_DIR)


gecho : gecho.cc
	$(CCC) $(CFLAGS) -o gecho gecho.cc  $(OBJS) $(LDFLAGS) 
	mv gecho $(OUTPUT_DIR)

g2Ptree : g2Ptree.cc ngraph.hpp ngraph_weak.hpp \
    ngraph_conductance.hpp
	$(CCC) $(CFLAGS) -o g2Ptree g2Ptree.cc  $(OBJS) $(LDFLAGS) 
	mv g2Ptree $(OUTPUT_DIR)

g2Pmetric_level : g2Pmetric_level.cc ngraph.hpp ngraph_weak.hpp \
    ngraph_conductance.hpp
	$(CCC) $(CFLAGS) -o g2Pmetric_level g2Pmetric_level.cc  $(OBJS) $(LDFLAGS) 
	mv g2Pmetric_level $(OUTPUT_DIR)

g2Pmatrix : g2Pmatrix.cc ngraph.hpp ngraph_weak.hpp \
    ngraph_conductance.hpp
	$(CCC) $(CFLAGS) -o g2Pmatrix g2Pmatrix.cc  $(OBJS) $(LDFLAGS) 
	mv g2Pmatrix $(OUTPUT_DIR)


g2cug : g2cug.cc
	$(CCC) $(CFLAGS) -o g2cug g2cug.cc  $(OBJS) $(LDFLAGS) 
	mv g2cug $(OUTPUT_DIR)


adj2g : adj2g.cc
	$(CCC) $(CFLAGS) -o adj2g adj2g.cc  $(OBJS) $(LDFLAGS) 
	mv adj2g $(OUTPUT_DIR)

g2Pmetric : g2Pmetric.cc
	$(CCC) $(CFLAGS) -o g2Pmetric g2Pmetric.cc  $(OBJS) $(LDFLAGS) 
	mv g2Pmetric $(OUTPUT_DIR)

g2degfreq : g2degfreq.cc
	$(CCC) $(CFLAGS) -o g2degfreq g2degfreq.cc  $(OBJS) $(LDFLAGS) 
	mv g2degfreq $(OUTPUT_DIR)

g2weak : g2weak.cc
	$(CCC) $(CFLAGS) -o g2weak g2weak.cc  $(OBJS) $(LDFLAGS) 
	mv g2weak $(OUTPUT_DIR)

ifreq : ifreq.cc
	$(CCC) $(CFLAGS) -o ifreq ifreq.cc  $(OBJS) $(LDFLAGS) 
	mv ifreq $(OUTPUT_DIR)

g2ug : g2ug.cc
	$(CCC) $(CFLAGS) -o g2ug g2ug.cc  $(OBJS) $(LDFLAGS) 
	mv g2ug $(OUTPUT_DIR)

g2scc : g2scc.cc
	$(CCC) $(CFLAGS) -o g2scc g2scc.cc  $(OBJS) $(LDFLAGS) 
	mv g2scc $(OUTPUT_DIR)


gtranspose : gtranspose.cc
	$(CCC) $(CFLAGS) -o gtranspose gtranspose.cc  $(OBJS) $(LDFLAGS) 
	mv gtranspose $(OUTPUT_DIR)


mi0_2mi : mi0_2mi.cc 
	$(CCC) $(CFLAGS) -o mi0_2mi mi0_2mi.cc  $(OBJS) $(LDFLAGS) 
	mv mi0_2mi $(OUTPUT_DIR)

gpagerank : gpagerank.cc pagerank.hpp  ngraph.hpp
	$(CCC) $(CFLAGS) -o gpagerank gpagerank.cc  $(OBJS) $(LDFLAGS) 
	mv gpagerank $(OUTPUT_DIR)

gintersect : gintersect.cc
	$(CCC) $(CFLAGS) -o gintersect gintersect.cc  $(OBJS) $(LDFLAGS) 
	mv gintersect $(OUTPUT_DIR)

gminus : gminus.cc
	$(CCC) $(CFLAGS) -o gminus gminus.cc  $(OBJS) $(LDFLAGS) 
	mv gminus $(OUTPUT_DIR)

v2t : v2t.cc
	$(CCC) $(CFLAGS) -o v2t v2t.cc  $(OBJS) $(LDFLAGS) 
	mv v2t $(OUTPUT_DIR)

gunion : gunion.cc
	$(CCC) $(CFLAGS) -o gunion gunion.cc  $(OBJS) $(LDFLAGS) 
	mv gunion $(OUTPUT_DIR)

gcompact : gcompact.cc
	$(CCC) $(CFLAGS) -o gcompact gcompact.cc  $(OBJS) $(LDFLAGS) 
	mv gcompact $(OUTPUT_DIR)

k2t : k2t.cc
	$(CCC) $(CFLAGS) -o k2t k2t.cc  $(OBJS) $(LDFLAGS) 
	mv k2t $(OUTPUT_DIR)

g2v : g2v.cc
	$(CCC) $(CFLAGS) -o g2v g2v.cc  $(OBJS) $(LDFLAGS) 
	mv g2v $(OUTPUT_DIR)

gcluster_coeff : gcluster_coeff.cc
	$(CCC) $(CFLAGS) -o gcluster_coeff gcluster_coeff.cc  $(OBJS) $(LDFLAGS) 
	mv gcluster_coeff $(OUTPUT_DIR)

g2udot : g2udot.cc
	$(CCC) $(CFLAGS) -o g2udot g2udot.cc  $(OBJS) $(LDFLAGS) 
	mv g2udot $(OUTPUT_DIR)

g2dot : g2dot.cc
	$(CCC) $(CFLAGS) -o g2dot g2dot.cc  $(OBJS) $(LDFLAGS) 
	mv g2dot $(OUTPUT_DIR)

g2tgf : g2tgf.cc
	$(CCC) $(CFLAGS) -o g2tgf g2tgf.cc  $(OBJS) $(LDFLAGS) 
	mv g2tgf $(OUTPUT_DIR)

g2g0 : g2g0.cc
	$(CCC) $(CFLAGS) -o g2g0 g2g0.cc  $(OBJS) $(LDFLAGS) 
	mv g2g0 $(OUTPUT_DIR)



mi2mt : mi2mt.cc
	$(CCC) $(CFLAGS) -o mi2mt mi2mt.cc  $(OBJS) $(LDFLAGS) 
	mv mi2mt $(OUTPUT_DIR)



mtx2g : mtx2g.c mmio.c
	$(CC) $(CFLAGS) -o mtx2g mtx2g.c mmio.c $(OBJS) $(LDFLAGS) 
	mv mtx2g $(OUTPUT_DIR)

gt2pajek : gt2pajek.cc ngraph.hpp
	$(CCC) $(CFLAGS) -o gt2pajek gt2pajek.cc $(OBJS) $(LDFLAGS) 
	mv gt2pajek $(OUTPUT_DIR)


gremove_v : gremove_v.cc ngraph.hpp
	$(CCC) $(CFLAGS) -o gremove_v gremove_v.cc $(OBJS) $(LDFLAGS) 
	mv gremove_v $(OUTPUT_DIR)


gsubgraph_v : gsubgraph_v.cc ngraph.hpp
	$(CCC) $(CFLAGS) -o gsubgraph_v gsubgraph_v.cc $(OBJS) $(LDFLAGS) 
	mv gsubgraph_v $(OUTPUT_DIR)

gprune : gprune.cc ngraph.hpp
	$(CCC) $(CFLAGS) -o gprune gprune.cc $(OBJS) $(LDFLAGS) 
	mv gprune $(OUTPUT_DIR)

vt2 : vt2.cc ngraph.hpp
	$(CCC) $(CFLAGS) -o vt2 vt2.cc $(OBJS) $(LDFLAGS) 
	mv vt2 $(OUTPUT_DIR)

g2degree : g2degree.cc ngraph.hpp
	$(CCC) $(CFLAGS) -o g2degree g2degree.cc $(OBJS) $(LDFLAGS) 
	mv g2degree $(OUTPUT_DIR)

gcomponents : gcomponents.cc ngraph.hpp
	$(CCC) $(CFLAGS) -o gcomponents gcomponents.cc $(OBJS) $(LDFLAGS) 
	mv gcomponents $(OUTPUT_DIR)

gnode_attack : gnode_attack.cc ngraph.hpp
	$(CCC) $(CFLAGS) -o gnode_attack gnode_attack.cc $(OBJS) $(LDFLAGS) 
	mv gnode_attack $(OUTPUT_DIR)

wipe:
	rm *.o  $(EXECS)
	

clean:
	rm *.o  
