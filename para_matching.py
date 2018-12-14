import numpy as np
import sharedmem, time
from readsubhalo import *

def readfrom_mb2(dirno,dirname2):
    
    save_path='/physics2/yingzhac/match/subfind/'
    sg_pos=np.load(save_path+'mb2_pos_'+dirno+'.npy')
    sg_mass=np.load(save_path+'mb2_mass_'+dirno+'.npy')
    sg_r=np.load(save_path+'r_mb2_'+dirno+'.npy')
    return sg_pos, sg_mass, sg_r

def readfrom_rockstar(listno,dirname_rs):
    
    save_path='/physics2/yingzhac/match/rockstar/'
    sg_pos=np.load(save_path+'rs_pos_'+listno+'.npy')
    sg_mass=np.load(save_path+'rs_mass_'+listno+'.npy')
    sg_r=np.load(save_path+'r_rs_'+listno+'.npy')
    return sg_pos, sg_mass, sg_r

def findis(cent1,cent2,boxlen1):

    permax = boxlen1/2.0
    Rdiff = cent1 - cent2
    Rmask = 1 - (np.abs(Rdiff/permax)).astype(np.int32)
    Roffset = np.ma.masked_array((-1)*boxlen1*((Rdiff/permax).astype(np.int32)),Rmask)
    Rdiff = np.array(Rdiff + Roffset.filled(0)).copy()
    Rmeas = pow(Rdiff,2.0)
    Rmeas3d = Rmeas[:,0] + Rmeas[:,1] + Rmeas[:,2]
    Rmeas3d = pow(Rmeas3d,1.0/2.0)
    
    return Rdiff, Rmeas3d

def match_rs_ids(sgpos_rs,sgmass_rs,sgr_rs,sgids_rs,sgpos_mb2,sgmass_mb2,sgr_mb2,sgids_mb2,mratio,mbound,rbound):

    global sgmatch_ids_rs_to_mb2   
    
    sgids_mb2_chk = sgids_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgids_rs_chk = sgids_rs[(sgmass_rs >= 1.0*mbound)]
    sgr_mb2_chk = sgr_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgr_rs_chk = sgr_rs[(sgmass_rs >= 1.0*mbound)]
    sgmass_mb2_chk = sgmass_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgmass_rs_chk = sgmass_rs[(sgmass_rs >= 1.0*mbound)]
    sgpos_mb2_chk = sgpos_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgpos_rs_chk = sgpos_rs[(sgmass_rs >= 1.0*mbound)]
    
    
    #print 'number of rs to match:', len(sgmass_rs_chk)
    #print 'number of mb2 to match:', len(sgmass_mb2_chk)
    for i in range(0,len(sgids_rs_chk)):
        fchk = ((sgmass_mb2_chk >= mratio*sgmass_rs_chk[i]) & (sgmass_mb2_chk*mratio <= sgmass_rs_chk[i])) 
        fpos_mb2 = sgpos_mb2_chk[fchk]
        fids_mb2 = sgids_mb2_chk[fchk]
        fRdiff, fRmeas3d = findis(fpos_mb2,sgpos_rs_chk[i],100.0)
        rchk = (fRmeas3d<=rbound*sgr_rs_chk[i])
        fRmeas3d=fRmeas3d[rchk]
        if (fchk.sum() > 0) and (rchk.sum()>0):
            #print sgids_rs_chk[i],sgids_mb2_chk[fchk][rchk][fRmeas3d.argmin()]
            sgmatch_ids_rs_to_mb2[sgids_rs_chk[i]] = (sgids_mb2_chk[fchk][rchk])[fRmeas3d.argmin()]
        
    
        #print i, len(sgpos_rs_chk)
    return  0

def match_mb2_ids(sgpos_rs,sgmass_rs,sgr_rs,sgids_rs,sgpos_mb2,sgmass_mb2,sgr_mb2,sgids_mb2,mratio,mbound,rbound):

    global sgmatch_ids_mb2_to_rs   
    
    sgids_mb2_chk = sgids_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgids_rs_chk = sgids_rs[(sgmass_rs >= 1.0*mbound)]
    sgr_mb2_chk = sgr_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgr_rs_chk = sgr_rs[(sgmass_rs >= 1.0*mbound)]
    sgmass_mb2_chk = sgmass_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgmass_rs_chk = sgmass_rs[(sgmass_rs >= 1.0*mbound)]
    sgpos_mb2_chk = sgpos_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgpos_rs_chk = sgpos_rs[(sgmass_rs >= 1.0*mbound)]
    
    
    #print 'number of rs to match:', len(sgmass_rs_chk)
    #print 'number of mb2 to match:', len(sgmass_mb2_chk)
    
    
    for i in range(0,len(sgids_mb2_chk)):
        fchk = ((sgmass_rs_chk >= mratio*sgmass_mb2_chk[i]) & (sgmass_rs_chk*mratio <= sgmass_mb2_chk[i]))
        fpos_rs = sgpos_rs_chk[fchk]
        fids_rs = sgids_rs_chk[fchk]
        fRdiff, fRmeas3d = findis(fpos_rs,sgpos_mb2_chk[i],100.0) 
        rchk = (fRmeas3d<=rbound*sgr_mb2_chk[i])
        fRmeas3d=fRmeas3d[rchk]
        if (fchk.sum() > 0) and (rchk.sum()>0):
            sgmatch_ids_mb2_to_rs[sgids_mb2_chk[i]] = (sgids_rs_chk[fchk][rchk])[fRmeas3d.argmin()]
        #print i, len(sgpos_mb2_chk)
    
    return  0

time1=time.time()
dirname_mb2 = '/physics/yfeng1/mb2'
dirname_rs = '/physics2/vat/merge-trees/constress_mb2_test_z1/cons_output'
fmb2_dirs = ('085','083','079','073','068','063','058','050','039')
rs_dirs=(36,34,30,24,19,14,9,4,0)

no_all=np.zeros(9,dtype='int')
no_matched=np.zeros(9,dtype='int')
eta=np.zeros(9)

logmbound=10
print 'logm=',logmbound
for fi_mb in range(8,9):
    dirno_mb2 = fmb2_dirs[fi_mb]
    listno_rs = str(rs_dirs[fi_mb])
    print fi_mb, dirno_mb2, listno_rs

    sgpos_mb2, sgmass_mb2, sgr_mb2= readfrom_mb2(dirno_mb2,dirname_mb2)
    print 'mb2 done'
    sgpos_rs, sgmass_rs, sgr_rs  = readfrom_rockstar(listno_rs,dirname_rs)
    print 'rockstar done'


    mratio=0.5
    rbound=10
    #logmbound=9.5
    mbound=10**logmbound
    rslen=len(sgmass_rs)
    mb2len=len(sgmass_mb2)
    print rslen,mb2len
    rlen=rslen/16
    mlen=mb2len/16
    mediate_path='/physics2/yingzhac/trees/new_results/mediate/'

    sgmatch_ids_rs_to_mb2=np.zeros(rslen,dtype=int)
    sgmatch_ids_mb2_to_rs=np.zeros(mb2len,dtype=int)
    sgmatch_flag=np.zeros(rslen,dtype=bool)

    np.save('begin_'+dirno_mb2+'_'+str(logmbound)+'_'+'.npy',mratio)

    sgids_mb2 = np.arange(0,len(sgmass_mb2))
    sgids_rs = np.arange(0,len(sgmass_rs))

    gnp=16
    slices1 = [slice(i,i+1) for i in range(0,gnp)]
    with sharedmem.Pool() as pool:
        def work(slice):
            si1 = slice.start
            sj1 = slice.stop
            mass_rs = sgmass_rs[si1*rlen : sj1*rlen]
            pos_rs = sgpos_rs[si1*rlen : sj1*rlen]
            r_rs = sgr_rs[si1*rlen : sj1*rlen]
            ids_rs = sgids_rs[si1*rlen : sj1*rlen]

            mass_mb2 = sgmass_mb2[si1*mlen : sj1*mlen]
            pos_mb2 = sgpos_mb2[si1*mlen : sj1*mlen]
            r_mb2 = sgr_mb2[si1*mlen : sj1*mlen]
            ids_mb2 = sgids_mb2[si1*mlen : sj1*mlen]


            if (si1 == (gnp - 1)):
                mass_rs = sgmass_rs[si1*rlen : rslen]
                pos_rs = sgpos_rs[si1*rlen : rslen]
                r_rs = sgr_rs[si1*rlen : rslen]
                ids_rs = sgids_rs[si1*rlen : rslen]

                mass_mb2 = sgmass_mb2[si1*mlen : mb2len]
                pos_mb2 = sgpos_mb2[si1*mlen : mb2len]
                r_mb2 = sgr_mb2[si1*mlen : mb2len]
                ids_mb2 = sgids_mb2[si1*mlen : mb2len]

            ids_rs_to_mb2_ij = match_rs_ids(pos_rs,mass_rs,r_rs,ids_rs,sgpos_mb2,sgmass_mb2,sgr_mb2,sgids_mb2,mratio,mbound,rbound)
            np.save(mediate_path+'sgmatch_ids_rs_to_mb2_'+dirno_mb2+'_'+str(logmbound)+str(si1)+'.npy',sgmatch_ids_rs_to_mb2)
            """sgmatch_ids_rs_to_mb2[si1*rlen : sj1*rlen]=ids_rs_to_mb2_ij
            if (si1 == (gnp - 1)):
                sgmatch_ids_rs_to_mb2[si1*rlen : rslen]=ids_rs_to_mb2_ij"""

            ids_mb2_to_rs_ij = match_mb2_ids(sgpos_rs,sgmass_rs,sgr_rs,sgids_rs,pos_mb2,mass_mb2,r_mb2,ids_mb2,mratio,mbound,rbound)
            np.save(mediate_path+'sgmatch_ids_mb2_to_rs_'+dirno_mb2+'_'+str(logmbound)+str(si1)+'.npy',sgmatch_ids_mb2_to_rs)
            """sgmatch_ids_mb2_to_rs[si1*mlen : sj1*mlen]=ids_mb2_to_rs_ij
            if (si1 == (gnp - 1)):
                sgmatch_ids_mb2_to_rs[si1*mlen : mb2len]=ids_mb2_to_rs_ij"""

        pool.map(work,slices1)


    for si1 in range(16):
        sj1 = si1 + 1
        ids_rs_to_mb2_ij=np.load(mediate_path+'sgmatch_ids_rs_to_mb2_'+dirno_mb2+'_'+str(logmbound)+str(si1)+'.npy')
        if (si1 == (gnp - 1)):
            sgmatch_ids_rs_to_mb2[si1*rlen : rslen]=ids_rs_to_mb2_ij[si1*rlen : rslen]
        else:
            sgmatch_ids_rs_to_mb2[si1*rlen : sj1*rlen]=ids_rs_to_mb2_ij[si1*rlen : sj1*rlen]


        ids_mb2_to_rs_ij=np.load(mediate_path+'sgmatch_ids_mb2_to_rs_'+dirno_mb2+'_'+str(logmbound)+str(si1)+'.npy')
        if (si1 == (gnp - 1)):
            sgmatch_ids_mb2_to_rs[si1*mlen : mb2len]=ids_mb2_to_rs_ij[si1*mlen : mb2len]
        else:
            sgmatch_ids_mb2_to_rs[si1*mlen : sj1*mlen]=ids_mb2_to_rs_ij[si1*mlen : sj1*mlen]



    sgids_mb2_chk = sgids_mb2[(sgmass_mb2 >= mratio*mbound)]
    sgids_rs_chk = sgids_rs[(sgmass_rs >= 1.0*mbound)]
    sgmatch_flag = np.zeros(len(sgpos_rs))   
    sgmatch_flag_fchk = sgmatch_flag[(sgmass_rs >= 1.0*mbound)]
    for i in range(0,len(sgmatch_flag_fchk)):
        if (sgids_rs_chk[i] == np.uint32(sgmatch_ids_mb2_to_rs[np.uint32(sgmatch_ids_rs_to_mb2[sgids_rs_chk[i]])])):
            sgmatch_flag[sgids_rs_chk[i]] = 1


    ffl = sgmatch_flag[(sgmass_rs >= 1.0*mbound)]
    print 'Total:',len(ffl)
    no_all[fi_mb]=len(ffl)
    print 'Successfully matched:',len(ffl[ffl == 1])
    no_matched[fi_mb]=len(ffl[ffl == 1])
    eff=0
    if len(ffl)>0:
        eff=len(ffl[ffl == 1])*100.0/len(ffl)
    print eff, "fraction"
    eta[fi_mb]=eff

    save_path='/physics2/yingzhac/summer2018/para_matching/'
    np.save(save_path+'eff_'+dirno_mb2+'_'+str(logmbound)+'.npy',eff)
    np.save(save_path+'sgmatch_ids_rs_to_mb2_'+dirno_mb2+'_'+str(logmbound)+'.npy',sgmatch_ids_rs_to_mb2)
    np.save(save_path+'sgmatch_flag_'+dirno_mb2+'_'+str(logmbound)+'.npy',sgmatch_flag)


    time2=time.time()

    print 'Time used: ',time2-time1



np.save(save_path+'no_all_'+str(logmbound)+'.npy',no_all)
np.save(save_path+'no_matched_'+str(logmbound)+'.npy',no_matched)
np.save(save_path+'eta_'+str(logmbound)+'.npy',eta)











