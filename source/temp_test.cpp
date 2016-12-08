/*
 *  Test file
 */

#include reaction.hpp
#include model.hpp
using namespace std;

int main(){
    
    model test_model= new model
    
    //her1 monomer
    reaction ph1_synthesis= new reaction(ph1,delay_ph1,mh1,ph1);
    reaction ph1_degradation= new reaction(pd1,0,ph1,ph1);
    reaction ph1_dissociation= new reaction(ddh11,0,ph11,new vector(ph1,ph11));
    reaction ph1_association= new reaction(dah11,0,new vector(ph1,ph1),new vector(ph1,ph11));
    
    //deltaC monomer
    reaction pd_synthesis= new reaction(psd,delay_pd,md,pd);
    reaction pd_degradatino= new reaction(pdd,0,pd,pd);
    
    //mesp1 monomer
    reaction mespa_synthesis= new reaction(psm1,depay_pm1,mm1,pm1);
    reaction mepsa_degradation= new reaction(pdm1,0,pm1,pm1);
    reaction mespa_dissociation= new reaction(dd11,0,pm11,new vector(pm1,pm11));
    reaction mespa_dissociation2= new reaction(dd12,0,pm12,new vector(pm1,pm12));
    reaction mespa_dissociation3= new reaction(dd22,0,pm22,new vector(pm1,pm22));
    reaction mespa_association= new reaction(dam11,0,new vector(pm1,pm1),new vector(pm1,pm11));
    reaction mespa_association2= new reaction(dam12,0,new vector(pm1,pm2),new vector(pm1,pm12));
    reaction mespa_association3= new reaction(dam22,0,new vector(pm2,pm2),new vector(pm1,pm22));

    //mepsb monomer
    reaction mespb_synthesis= new reaction(psm2,depay_pm2,mm2,pm2);
    reaction mepsb_degradation= new reaction(pdm2,0,pm2,pm2);
    reaction mespb_dissociation= new reaction(dd11,0,pm11,new vector(pm2,pm11));
    reaction mespb_dissociation2= new reaction(dd12,0,pm12,new vector(pm2,pm11));
    reaction mespb_dissociation3= new reaction(dd22,0,pm22,new vector(pm2,pm11));
    reaction mespb_association= new reaction(dam11,0,new vector(pm1,pm1),new vector(pm2,pm11));
    reaction mespb_association2= new reaction(dam12,0,new vector(pm1,pm2),new vector(pm2,pm12));
    reaction mespb_association3= new reaction(dam22,0,new vector(pm2,pm2),new vector(pm2,pm22));

    
    //part of those correlates to reactions above.
    //her1-her1 dimer
    reaction ph11_degradation= new reaction(pd11,0,ph11,ph11);
    //mespa-mespa dimer
    reaction pm11_degradation= new reaction(pdm11,0,pm11,pm11);
    //mespa-mespb dimer
    reaction pm12_degradation= new reaction(pdm12,0,pm12,pm12);
    //mespb-mespb dimer
    reaction pm22_degradation= new reaction(pdm22,0,pm22,pm22);
    
    //her1 mRNA
    reaction mh1_synthesis= new reation(msp1,delay_mh1,new vector(pd,ph11),mh1);
    reaction mh1_degradation= new reaction(mdh1,0,mh1,mh1);
    
    //deltaC mRNA
    reaction md_synthesis= new reation(msd,delay_md,ph11,md);
    reaction md_degradation= new reaction(mdd,0,md,md);
    
    //mespa mRNA
    reaction mm1_synthesis= new reation(msm1,delay_mm1,new vector(pd,pm22,ph11),mm1);
    reaction mm1_degradation= new reaction(mdm1,0,mm1,mm1);
    
    //mespb mRNA
    reaction mm2_synthesis= new reation(msm2,delay_mm2,new vector(pd,pm11,pm12,pm22),mm2);
    reaction mm2_degradation= new reaction(mdm2,0,mm2,mm2);
}
