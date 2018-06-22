/*********************************************************************
 ** Stochastic Ranking Evolution Strategy                           **
 ** with MPI                                                        **
 ** (miu,lambda)-Evolution Strategy                                 **
 **                                                                 **
 ** For ACADEMIC RESEARCH, this is licensed with GPL license        **
 ** For COMMERCIAL ACTIVITIES, please contact the authors           **
 **                                                                 **
 ** Copyright (C) 2005 Xinglai Ji (jix1@ornl.gov)                   **
 **                                                                 **
 ** This program is distributed in the hope that it will be useful, **
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of  **
 ** MERCHANTABILITY of FITNESS FOR A PARTICULAR PURPOSE. See the    **
 ** GNU General Public License for more details.                    **
 **                                                                 **
 ** You should have received a copy of the GNU General Public       **
 ** License along with is program; if not, write to the Free        **
 ** Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, **
 ** MA 02111-1307, USA.                                             **
 **                                                                 **
 ** Author: Xinglai Ji (jix1@ornl.gov)                              **
 ** Date:   Apr 5, 2005;                                            **
 ** Organization: Oak Ridge National Laboratory                     **
 ** Reference:                                                      **
 **   1. Thomas P. Runarsson and Xin Yao. 2000. Stochastic Ranking  **
 **      for Constrained Evolutionary Optimization. 4(3):284-294.   **
 **      http://cerium.raunvis.hi.is/~tpr/software/sres/            **
 **   2. Thomas Philip Runarsson and Xin Yao. 2005. Search Biases   **
 **      in Constrained Evolutionary Optimization. IEEE             **
 **      Transactions on Systems, Man and Cybernetics -- Part C:    **
 **      Applications and Reviews. 35(2):233-243.                   **
 **   3. MPICH                                                      **
 **      http://www-unix.mcs.anl.gov/mpi/mpich/                     **
 *********************************************************************/

#if defined(MPI)
	#undef MPI
	#include <mpi.h>
	#define MPI
#endif
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sharefunc.hpp"
#include "ESSRSort.hpp"
#include "ESES.hpp"

#include "../source/io.hpp"

extern int printing_precision; // Declared in main.cpp

/*********************************************************************
 ** Initialize: parameters,populations and random seed              **
 ** ESInitial( argc, argv,                                          **
 **            seed, param,trsfm, fg,es, constraint,                **
 **            dim,ub,lb,miu,lambda,gen,                            **
 **              gamma, alpha, varphi, retry, population, stats)    **
 ** seed: random seed, usually esDefSeed=0 (pid*time)               **
 ** outseed: seed value assigned , for next use                     **
 ** param: point to parameter                                       **
 ** trsfm: to transform sp/op                                       **
 ** fg: functions of fitness and constraints                        **
 ** es: ES process, esDefESPlus/esDefESSlash                        **
 ** constraint: number of constraints                               **
 ** dim: dimension/number of genes in genome                        **
 ** ub[dim]: up bounds                                              **
 ** lb[dim]: low bounds                                             **
 ** miu: parent/population size                                     **
 ** lambda: offsping/population size                                **
 ** gen: number of generations                                      **
 ** gamma: usually esDefGamma=0.85                                  **
 ** alpha: usually esDefAlpha=0.2                                   **
 ** chi: chi = 1/2n +1/2sqrt(n)                                     **
 ** varphi: = sqrt((2/chi)*log((1/alpha)*(exp(varphi^2*chi/2)       **
 **                  -(1-alpha))))                                  **
 **         expected rate of convergence                            **
 ** retry: retry times to check bounds                              **
 ** tau: learning rates: tau = varphi/(sqrt(2*sqrt(dim)))           **
 ** tar_: learning rates: tau_ = varphi((sqrt(2*dim)                **
 ** population: point this population                               **
 ** stats: statistics for each generation                           **
 **                                                                 **
 ** Initialize MPI                                                  **
 ** processors >=2, print seed if master                            **
 **                                                                 **
 ** ESDeInitial(param,population,stats)                             **
 ** free param and population                                       **
 ** finalize MPI                                                    **
 *********************************************************************/
void ESInitial(/*int *argc, char ***argv,   */\
               unsigned int seed, ESParameter ** param,ESfcnTrsfm *trsfm,  \
               ESfcnFG fg, int es, int constraint, int dim, double* ub,   \
               double *lb, int miu, int lambda, int gen,  \
               double gamma, double alpha, double varphi, int retry,  \
               ESPopulation ** population, ESStatistics **stats)
{
  unsigned int outseed;
  int myid, numprocs;

  //MPI_Init(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if(numprocs < 2)
  {
    printf("Requiring at least 2 processes!\n");
    exit(1);
  }

  ShareSeed(seed, &outseed);
  ESInitialParam(param, trsfm, fg, es, outseed,constraint, dim, ub, lb,   \
                 miu, lambda, gen, gamma, alpha, varphi, retry);
  ESInitialPopulation(population, (*param));
  ESInitialStat(stats, (*population), (*param));

  /*if(myid == 0)
  {
    printf("\n========\nseed = %u\n========\n", outseed);
    fflush(nullptr);
  }*/

  return;
}
void ESDeInitial(ESParameter *param, ESPopulation *population,   \
                 ESStatistics *stats)
{
  ESDeInitialPopulation(population, param);
  ESDeInitialStat(stats);
  ESDeInitialParam(param);

  MPI_Finalize();
  return;
}

/*********************************************************************
 ** initialize parameters                                           **
 ** ESInitialParam(param, trsfm, fg,constraint,                     **
 **                dim,ub,lb,miu,lambda,gen)                        **
 ** param: point to parameter                                       **
 ** trsfm: to transform sp/op                                       **
 ** fg: functions of fitness and constraints                        **
 ** es: ES process, esDefESPlus/esDefESSlash                        **
 ** seed: reserve seed for next use                                 **
 ** constraint: number of constraints                               **
 ** dim: dimension/number of genes in genome                        **
 ** ub[dim]: up bounds                                              **
 ** lb[dim]: low bounds                                             **
 ** spb[dim]: bounds on sp , spb = (ub-lb)/sqrt(dim)                **
 ** miu: parent/population size                                     **
 ** lambda: offsping/population size                                **
 ** gen: number of generations                                      **
 ** gamma: usually esDefGamma=0.85                                  **
 ** alpha: usually esDefAlpha=0.2                                   **
 ** chi: chi = 1/2n +1/2sqrt(n)                                     **
 ** varphi: = sqrt((2/chi)*log((1/alpha)*(exp(varphi^2*chi/2)       **
 **                  -(1-alpha))))                                  **
 **         expected rate of convergence                            **
 ** retry: retry times to check bounds                              **
 ** tau: learning rates: tau = varphi/(sqrt(2*sqrt(dim)))           **
 ** tar_: learning rates: tau_ = varphi((sqrt(2*dim)                **
 **                                                                 **
 ** ESDeInitialParam(param)                                         **
 ** free param                                                      **
 *********************************************************************/
void ESInitialParam(ESParameter **param,ESfcnTrsfm *trsfm,  \
                    ESfcnFG fg, int es, unsigned int seed,  \
                    int constraint, int dim,  double *ub, double *lb,   \
                    int miu, int lambda, int gen,  \
                    double gamma, double alpha,  \
                    double varphi, int retry)
{
  int i;

  (*param) = (ESParameter *)ShareMallocM1c(sizeof(ESParameter));
  (*param)->trsfm = nullptr;
  (*param)->fg = nullptr;
  (*param)->ub = nullptr;
  (*param)->lb = nullptr;
  (*param)->spb = nullptr;

  (*param)->spb = ShareMallocM1d(dim);
  for(i=0; i<dim; i++)
    (*param)->spb[i] = (ub[i] - lb[i])/sqrt(dim);

  (*param)->trsfm = trsfm;
  (*param)->fg = fg;
  (*param)->es = es;
  (*param)->seed = seed;
  (*param)->constraint = constraint;
  (*param)->dim = dim;
  (*param)->ub = ub;
  (*param)->lb = lb;
  (*param)->miu = miu;
  (*param)->lambda = lambda;
  (*param)->gen = gen;
  (*param)->gamma = gamma;
  (*param)->alpha = alpha;
  (*param)->varphi = varphi;
  (*param)->retry = retry;

  if(es == esDefESSlash)
    (*param)->eslambda = lambda;
  else if(es == esDefESPlus)
    (*param)->eslambda = lambda + miu;
  else
    (*param)->eslambda = lambda;

  (*param)->chi = 1.0/(2*dim) + 1.0/(2*sqrt(dim));
  (*param)->varphi = sqrt((2.0/(*param)->chi)*log((1.0/alpha)    \
                  *(exp(varphi*varphi*(*param)->chi/2)-(1-alpha))));
  (*param)->tau = (*param)->varphi/(sqrt(2*sqrt(dim)));
  (*param)->tau_ = (*param)->varphi/(sqrt(2*dim));


  return;
}
void ESDeInitialParam(ESParameter *param)
{
  ShareFreeM1d(param->spb);
  ShareFreeM1c((char *)param);
  param = nullptr;
  return;
}

/*********************************************************************
 ** initialize population                                           **
 ** ESInitialPopulation(population,param)                           **
 ** population: point to this population                            **
 ** param: point to this parameter                                  **
 **   -> index: 0->eslambda-1                                       **
 **   -> individual[eslambda]                                       **
 **   -> fg(individual)                                             **
 **   -> f,phi                                                      **
 ** the initialization is looked as first generation                **
 **                                                                 **
 ** ESDeInitialPopulation(population, param)                        **
 ** free population                                                 **
 *********************************************************************/
void ESInitialPopulation(ESPopulation **population, ESParameter *param)
{
  int i;
  int eslambda;

  eslambda = param->eslambda;

  (*population) = (ESPopulation *)ShareMallocM1c(sizeof(ESPopulation));
  (*population)->member = nullptr;
  (*population)->f = nullptr;
  (*population)->phi = nullptr;
  (*population)->index = nullptr;

  (*population)->member = (ESIndividual **)  \
                   ShareMallocM1c(eslambda*sizeof(ESIndividual *));
  (*population)->f = ShareMallocM1d(eslambda);
  (*population)->phi = ShareMallocM1d(eslambda);

  (*population)->index = ShareMallocM1i(eslambda);

  for(i=0; i<eslambda; i++)
  {
    (*population)->member[i] = nullptr;
    ESInitialIndividual(&((*population)->member[i]), param);
    (*population)->index[i] = i;
    (*population)->f[i] = (*population)->member[i]->f;
    (*population)->phi[i] = (*population)->member[i]->phi;
  }

  return;
}
void ESDeInitialPopulation(ESPopulation *population, ESParameter *param)
{
  int i;
  int eslambda;

  eslambda = param->eslambda;

  for(i=0; i<eslambda; i++)
    ESDeInitialIndividual(population->member[i]);
  ShareFreeM1c((char *)(population->member));

  ShareFreeM1d(population->f);
  ShareFreeM1d(population->phi);
  ShareFreeM1i(population->index);
  ShareFreeM1c((char*)population);
  population = nullptr;

  return;
}


/*********************************************************************
 ** initialize individual                                           **
 ** ESInitialIndividual(indvdl, param)                              **
 ** to calculate f,g,and phi                                        **
 ** to initialize op and sp                                         **
 ** phi=sum{(g>0)^2}                                                **
 ** op = rand(lb, ub)                                               **
 ** sp = (ub - lb)/sqrt(dim)                                        **
 **                                                                 **
 **                                                                 **
 ** ESDeInitialIndividual(indvdl)                                   **
 ** free individual                                                 **
 **                                                                 **
 ** ESPrintOp(indvdl, param)                                        **
 ** print individual information, indvdl->op                        **
 ** ESPrintSp(indvdl, param)                                        **
 ** print individual information, indvdl->sp                        **
 *********************************************************************/
void ESInitialIndividual(ESIndividual **indvdl, ESParameter *param)
{
  int i;
  int dim;
  int constraint;
  ESfcnFG fg;
  double *ub, *lb;

  dim = param->dim;
  constraint = param->constraint;
  fg = param->fg;
  ub = param->ub;
  lb = param->lb;

  (*indvdl) = (ESIndividual *)ShareMallocM1c(sizeof(ESIndividual));
  (*indvdl)->op = nullptr;
  (*indvdl)->sp = nullptr;
  (*indvdl)->g = nullptr;

  (*indvdl)->op = ShareMallocM1d(dim);
  (*indvdl)->sp = ShareMallocM1d(dim);
  if (constraint > 0) {
    (*indvdl)->g = ShareMallocM1d(constraint);
  } else {
    (*indvdl)->g = nullptr;
  }

  for(i=0; i<dim; i++)
  {
    (*indvdl)->op[i] = ShareRand(lb[i], ub[i]);
    (*indvdl)->sp[i] = (ub[i] - lb[i])/sqrt(dim);
  }

  fg((*indvdl)->op, &((*indvdl)->f), (*indvdl)->g);
  (*indvdl)->phi = 0.0;
  for(i=0; i<constraint; i++)
  {
    if((*indvdl)->g[i] > 0.0)
      (*indvdl)->phi += ((*indvdl)->g[i])*((*indvdl)->g[i]);
  }

  return;
}
void ESDeInitialIndividual(ESIndividual *indvdl)
{
  ShareFreeM1d(indvdl->g);
  ShareFreeM1d(indvdl->op);
  ShareFreeM1d(indvdl->sp);
  ShareFreeM1c((char *)indvdl);
  indvdl = nullptr;

  return;
}
void ESPrintIndividual(ESIndividual *indvdl, ESParameter *param)
{
  return;
}
void ESPrintOp(ESIndividual *indvdl, ESParameter *param)
{
  int i = 0;
  int dim;
  ESfcnTrsfm *trsfm;

  trsfm = param->trsfm;
  dim = param->dim;

  if (trsfm == nullptr) {
    printf("%.*f", printing_precision, indvdl->op[0]);
    for (i=1; i<dim; i++) {
      printf(",%.*f", printing_precision, indvdl->op[i]);
    }
  } else {
    if (trsfm[0] == nullptr) {
      printf("%.*f", printing_precision, (trsfm[i])(indvdl->op[0]));
    } else {
      printf("%.*f", printing_precision, indvdl->op[0]);
    }
    for (i=1; i<dim; i++) {
      if (trsfm[i] == nullptr) {
        printf(",%.*f", printing_precision, indvdl->op[i]);
      } else {
        printf(",%.*f", printing_precision, (trsfm[i])(indvdl->op[i]));
      }
    }
  }

  return;
}
void ESPrintSp(ESIndividual *indvdl, ESParameter *param)
{
  int i;
  int dim;
  ESfcnTrsfm *trsfm;

  trsfm = param->trsfm;
  dim = param->dim;
  if(trsfm == nullptr)
    for(i=0; i<dim; i++)
      printf("\t%f", indvdl->sp[i]);
  else
    for(i=0; i<dim; i++)
      if(trsfm[i] == nullptr)
        printf("\t%f", indvdl->sp[i]);
      else
        printf("\t%f", (trsfm[i])(indvdl->sp[i]));

  return;
}

/*********************************************************************
 ** copy a individual                                               **
 ** ESCopyIndividual(from, to, param)                               **
 *********************************************************************/
void ESCopyIndividual(ESIndividual *from, ESIndividual *to, ESParameter *param)
{
  int i;
  int dim;
  int constraint;

  dim = param->dim;
  constraint = param->constraint;

  for(i=0; i<dim; i++)
  {
    to->op[i] = from->op[i];
    to->sp[i] = from->sp[i];
  }
  for(i=0; i<constraint; i++)
    to->g[i] = from->g[i];
  to->f = from->f;
  to->phi = from->phi;

  return;
}

/*********************************************************************
 ** initialize statistics                                           **
 ** ESInitialStat(stats, population, param)                         **
 ** to intialize time, curgen, bestindvdl,thisbestindvdl            **
 ** not to do the first statistics                                  **
 ** to set dt, bestgen                                              **
 **                                                                 **
 ** ESDeInitialStat(stats)                                          **
 ** free statistics                                                 **
 *********************************************************************/
void ESInitialStat(ESStatistics **stats, ESPopulation *population,   \
                   ESParameter *param)
{
  (*stats) = (ESStatistics *)ShareMallocM1c(sizeof(ESStatistics));
  (*stats)->bestgen = 0;
  (*stats)->curgen = 0;
  (*stats)->bestindvdl = nullptr;
  (*stats)->thisbestindvdl = nullptr;
  (*stats)->dt = 0;
  time(&((*stats)->begintime));
  time(&((*stats)->nowtime));

  ESInitialIndividual(&((*stats)->bestindvdl), param);
  ESInitialIndividual(&((*stats)->thisbestindvdl), param);

/*********************************************************************
 ** dont do stat when initializing                                  **
 ** ESDoStat((*stats), population, param);                          **
 *********************************************************************/

  return;
}
void ESDeInitialStat(ESStatistics *stats)
{
  ESDeInitialIndividual(stats->bestindvdl);
  ESDeInitialIndividual(stats->thisbestindvdl);
  ShareFreeM1c((char*)stats);
  stats = nullptr;

  return;
}

/*********************************************************************
 ** do statistics                                                   **
 ** ESDoStat(stats, population, param)                              **
 ** to set nowtime, dt, curgen, bestgen, (this)bestindvdl           **
 ** to do statistics                                                **
 ** if there's no feasible best, do nothing                         **
 ** the initialization is looked as zero generation                 **
 **                                                                 **
 ** ESPrintStat(stats, param)                                       **
 ** print statistics information                                    **
 ** gen=,time=,dt=,bestgen=,bestfitness=,bestindividual=,           **
 *********************************************************************/
void ESDoStat(ESStatistics *stats, ESPopulation *population,   \
              ESParameter *param)
{
  int i;
  int eslambda;
  int flag,count;

  eslambda = param->eslambda;

  stats->curgen +=1;
  time(&(stats->nowtime));
  stats->dt = stats->nowtime - stats->begintime;

  flag = -1;
  count = 0;
  for(i=0; i<eslambda; i++)
  {
    if(ShareIsZero(population->phi[i])==shareDefTrue )
      count++;
    else
      continue;
    if(flag < 0)
    {
      flag = i;
      continue;
    }
    if(population->f[i] < population->f[flag])
      flag = i;
  }
  if(count <=0)
    return;

  ESCopyIndividual(population->member[flag], stats->thisbestindvdl, param);
  if( population->f[flag] <= stats->bestindvdl->f  \
      || ShareIsZero(stats->bestindvdl->phi)==shareDefFalse )
  {
    ESCopyIndividual(population->member[flag], stats->bestindvdl, param);
    stats->bestgen = stats->curgen;
  }

  return;
}

void ESPrintStat(ESStatistics *stats, ESParameter *param)
{

  printf("current generation: %d, best generation: %d, best fitness: %f\nbest individual: ",  \
          stats->curgen,stats->bestgen,stats->bestindvdl->f);
  ESPrintOp(stats->bestindvdl, param);
  printf("\n");
  /*printf("      variance:");
  ESPrintSp(stats->bestindvdl, param);
  printf("\n");*/
  fflush(nullptr);

  return;
}

/*********************************************************************
 ** stepwise evolution                                              **
 ** ESStep(population, param, stats, pf)                            **
 **                                                                 **
 ** Master:                                                         **
 ** -> Stochastic ranking -> sort population based on ranking index **
 ** -> send op to other processors  -> do statistics analysis on    **
 ** this generation -> print statistics information                 **
 ** Slave:                                                          **
 ** recalculate f/g/phi -> curgen+1                                 **
 *********************************************************************/
void ESStep(ESPopulation *population, ESParameter *param,   \
            ESStatistics *stats, double pf)
{
  int myid;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if(myid == 0)
  {
    ESSRSort(population->f, population->phi, pf, param->eslambda,   \
             param->eslambda, population->index);
    ESSortPopulation(population, param);

    ESSelectPopulation(population, param);

    ESMutate(population, param);

    ESDoStat(stats, population, param);

    ESPrintStat(stats, param);
  }
  else
  {
    ESMPIMutate(population, param);
    stats->curgen +=1;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return;
}

/*********************************************************************
 ** sort population based on Index by ESSRSort                      **
 ** ESSortPopulation(population, param)                             **
 *********************************************************************/
void ESSortPopulation(ESPopulation *population, ESParameter *param)
{
  int i;
  int eslambda;
  int *index;
  ESIndividual **oldmember, **newmember;

  eslambda = param->eslambda;
  oldmember = population->member;
  index = population->index;
  newmember = nullptr;
  newmember = (ESIndividual **)ShareMallocM1c(eslambda*sizeof(ESIndividual *));

  for(i=0; i<eslambda; i++)
    newmember[i] = oldmember[index[i]];

  for(i=0; i<eslambda; i++)
  {
    oldmember[i] = newmember[i];
    population->f[i] = newmember[i]->f;
    population->phi[i] = newmember[i]->phi;
    index[i] = i;
  }

  ShareFreeM1c((char *)newmember);
  newmember = nullptr;

  return;
}

/*********************************************************************
 ** select the next generation                                      **
 ** ESSelectPopulation(population, param)                           **
 ** select first miu offsprings to fill up the next generation      **
 ** miu -> lambda : 1..miu,1..miu,..,lambda                         **
 *********************************************************************/
void ESSelectPopulation(ESPopulation *population, ESParameter *param)
{
  int i,j;
  int miu, lambda, eslambda;

  miu = param->miu;
  lambda = param->lambda;
  eslambda = param->eslambda;

  for(i=miu,j=0; i<lambda; i++,j++)
  {
    if(j==miu)
      j = 0;
    ESCopyIndividual(population->member[j], population->member[i], param);
    population->f[i] = population->member[j]->f;
    population->phi[i] = population->member[j]->phi;
  }
  for(i=lambda, j=0; i<eslambda; i++,j++)
  {
    if(j==miu)
      j = 0;
    ESCopyIndividual(population->member[j], population->member[i], param);
    population->f[i] = population->member[j]->f;
    population->phi[i] = population->member[j]->phi;
  }

  return;
}

/*********************************************************************
 ** mutate                                                          **
 ** ESMutate(population, param)                                     **
 **                                                                 **
 ** sp_ : copy of sp                                                **
 ** op_ : copy of op                                                **
 ** update sp                                                       **
 ** traditional technique using exponential smoothing               **
 ** sp(1->miu-1) : unchanged                                        **
 ** sp(miu->lambda): sp = sp_*exp(tau_*N(0,1) + tau*Nj(0,1))        **
 **                  Nj : random number generated for each j        **
 ** check sp bound                                                  **
 ** if(sp > bound) then sp = bound                                  **
 ** differential variation                                          **
 ** op(1->miu-1) = op_ + gamma*(op_[1] - op_[i+1])                  **
 ** mutation                                                        **
 ** op(miu->lambda): op = op_ +sp * N(0,1)                          **
 ** check op bound                                                  **
 ** if(op > ub || op < lb) then try retry times                     **
 **                        op = op_ + sp * N(0,1)                   **
 ** if still not in bound then op = op_                             **
 ** exponential smoothing                                           **
 ** sp(miu->lambda): sp = sp_ + alpha * (sp - sp_)                  **
 **
 ** Master: send op to other processors                             **
 ** Slave:  re-calculate f/g/phi                                    **
 *********************************************************************/
void ESMutate(ESPopulation * population, ESParameter *param)
{
  int i, j, k, l;
  int miu, dim,lambda, constraint;
  double gamma, alpha;
  double tau, tau_;
  int retry;
  double randscalar;
  double *randvec;
  double *spb, *ub, *lb;
  ESIndividual *indvdl;
  double **sp_, **op_;
  double tmp;
  ESfcnFG fg;

  int numprocs,nummpi;
  MPI_Status status;
  double *gfphi;
  char strOK[] = "OK";
  int lenOK = 2;
  lenOK = strlen(strOK);

  randvec = nullptr;
  sp_ = nullptr;
  op_ = nullptr;

  miu = param->miu;
  lambda = param->lambda;
  constraint = param->constraint;
  gamma = param->gamma;
  alpha = param->alpha;
  tau = param->tau;
  tau_ = param->tau_;
  retry = param->retry;
  spb = param->spb;
  ub = param->ub;
  lb = param->lb;
  dim = param->dim;
  fg = param->fg;
  randvec = ShareMallocM1d(dim);
  sp_ = ShareMallocM2d(lambda, dim);
  op_ = ShareMallocM2d(lambda, dim);

  for(i=0; i<lambda; i++)
  {
    indvdl = population->member[i];
    for(j=0; j<dim; j++)
    {
      sp_[i][j] = indvdl->sp[j];
      op_[i][j] = indvdl->op[j];
    }
  }

  for(i=miu-1; i<lambda; i++)
  {
    randscalar = ShareNormalRand(0,1);
    ShareNormalRandVec(randvec, dim, 0, 1);
    indvdl = population->member[i];
    for(j=0; j<dim; j++)
    {
      tmp = indvdl->sp[j] *exp(tau_ *randscalar +tau*randvec[j]);
      if( tmp > spb[j] )
        tmp = spb[j];
      indvdl->sp[j] = tmp;
    }
  }

  for(i=0; i<miu-1; i++)
  {
    indvdl = population->member[i];
    for(j=0; j<dim; j++)
      indvdl->op[j] = indvdl->op[j] + gamma*(op_[0][j] - op_[i+1][j]);
  }
  for(i=miu-1; i<lambda; i++)
  {
    indvdl = population->member[i];
    for(j=0; j<dim; j++)
      indvdl->op[j] = indvdl->op[j] + indvdl->sp[j] * ShareNormalRand(0, 1);
  }

  for(i=0; i<lambda; i++)
  {
    indvdl = population->member[i];
    for(j=0; j<dim; j++)
    {
      tmp = indvdl->op[j];
      if(tmp > ub[j] || tmp < lb[j])
      {
        for(k=0; k<retry; k++)
        {
          tmp = op_[i][j] + indvdl->sp[j]*ShareNormalRand(0,1);
          if(!(tmp > ub[j] || tmp < lb[j]))
            break;
        }
        if(k >= retry)
          tmp = op_[i][j];
        indvdl->op[j] = tmp;
      }
    }
  }

  for(i=miu-1; i<lambda; i++)
  {
    indvdl = population->member[i];
    for(j=0; j<dim; j++)
      indvdl->sp[j] = sp_[i][j] + alpha *(indvdl->sp[j] - sp_[i][j]);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  for(i=0,j=1; i<lambda; i++,j++)
  {
    if(j==numprocs)
      j=1;
    MPI_Send(population->member[i]->op,dim,MPI_DOUBLE,j,i,MPI_COMM_WORLD);
  }
  gfphi = ShareMallocM1d(2+constraint);
  for(l=1; l<numprocs; l++)
  {
    for(i=0,j=1,nummpi=0; i<lambda; i++,j++)
    {
      if(j==numprocs)
        j=1;
      if(j==l)
        nummpi++;
    }
    if(nummpi<=0)
      break;
    MPI_Send(strOK,lenOK,MPI_BYTE,l,l,MPI_COMM_WORLD);
    for(i=0,j=1; i<lambda; i++,j++)
    {
      if(j==numprocs)
        j=1;
      if(j!=l)
        continue;
      MPI_Recv(gfphi,2+constraint,MPI_DOUBLE,j,i,MPI_COMM_WORLD,&status);
      indvdl = population->member[i];
      for(k=0;k<constraint;k++)
        indvdl->g[k] = gfphi[k];
      indvdl->f = gfphi[k++];
      indvdl->phi = gfphi[k++];
      population->f[i] = indvdl->f;
      population->phi[i] = indvdl->phi;
    }
  }
  ShareFreeM1d(gfphi);
  gfphi = nullptr;

  ShareFreeM1d(randvec);
  randvec = nullptr;
  ShareFreeM2d(sp_, lambda);
  sp_ = nullptr;
  ShareFreeM2d(op_, lambda);
  op_ = nullptr;

  return;
}

void ESMPIMutate(ESPopulation *population, ESParameter *param)
{
  int i,j,k,l;
  int lambda, dim, constraint;

  double *op, **gfphi;
  int nummpi;
  int myid, numprocs;
  MPI_Status status;
  char buf[10];
  char strOK[] = "OK";
  int lenOK = 2;
  lenOK = strlen(strOK);

  lambda = param->lambda;
  dim = param->dim;
  constraint = param->constraint;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  for(i=0,j=1,nummpi=0; i<lambda; i++,j++)
  {
    if(j==numprocs)
      j = 1;
    if(j==myid)
      nummpi++;
  }
  if(nummpi<=0)
    return;

  op = ShareMallocM1d(dim);
  gfphi = ShareMallocM2d(nummpi,2+constraint);

  for(i=0,j=1,l=0; i<lambda; i++,j++)
  {
    if(j==numprocs)
      j = 1;
    if(j!=myid)
      continue;
    MPI_Recv(op, dim, MPI_DOUBLE, 0,i,MPI_COMM_WORLD,&status);
    param->fg(op, &(gfphi[l][constraint]),gfphi[l]);
    gfphi[l][constraint+1] = 0.0;
    for(k=0;k<constraint;k++)
    {
      if(gfphi[l][k]>0.0)
        gfphi[l][constraint+1] += (gfphi[l][k]*gfphi[l][k]);
    }
    l++;
  }

  MPI_Recv(buf,lenOK,MPI_BYTE,0,myid,MPI_COMM_WORLD, &status);
  for(i=0,j=1,l=0; i<lambda; i++,j++)
  {
    if(j==numprocs)
      j = 1;
    if(j!=myid)
      continue;
    MPI_Send(gfphi[l], 2+constraint, MPI_DOUBLE, 0,i,MPI_COMM_WORLD);
    l++;
  }

  ShareFreeM1d(op);
  op = nullptr;
  ShareFreeM2d(gfphi,nummpi);
  gfphi = nullptr;

  return;
}

