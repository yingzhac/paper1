import numpy as np
import time


loaddir1='/physics2/yingzhac/trees/results/'
loaddir2='/physics2/yingzhac/paper1/tree/z=3/'
a=np.load('/home/yingzhac/fall2017/save/scale_factor.npy')



for ii in range(4):
    for jj in range(4):
        for kk in range(4):
            textt=[ii,jj,kk]
            np.savetxt('buiding_tree3_'+str(ii)+'.txt',textt)
            t1=time.time()
            treenpy=loaddir1+'tree_'+str(ii)+'_'+str(jj)+'_'+str(kk)+'.npy'
            tree=np.load(treenpy)
            scale=tree[:,0]
            t=tree[(scale<=0.625) & (scale>=0.25)]
            treenpy=loaddir2+'tree'+str(ii)+str(jj)+str(kk)+'.npy'
            np.save(treenpy,t)
            
            
            ntot=len(tree[tree[:,0]==0.94118])
            flag= (t[:,0]==0.625)
            begin=np.zeros(ntot,dtype=int)-1
            end=np.zeros(ntot,dtype=int)-1
            begin[0]=0
            i=0
            s=0
            while i<len(flag)-1:
                if (flag[i]==1) and (flag[i+1]==0):
                    s+=1
                if (flag[i]==0) and (flag[i+1]==1):
                    end[s-1]=i
                    begin[s]=i+1

                i+=1
                
            sca=a[(a<=0.625) & (a>=0.25)]
            lines=flag.sum()
            index=np.zeros((lines,25),dtype=int)-1
            red=np.zeros((lines,25))



            trno=0
            s=0
            for i in range(len(t)):
                if t[i,0]==0.625:
                    if i>end[trno] and end[trno]>=0:
                        trno+=1
                    rootid=t[i,28]
                    for j in range(25):
                        progid=rootid+j

                        find=np.argwhere(t[begin[trno]:end[trno],28]==progid)
                        if len(find>0):
                            rowno=begin[trno]+find[0,0]
                            if t[rowno,0]<>sca[j]:
                                break
                            index[s,j]=rowno
                    s+=1
                    
            indexnpy=loaddir2+'index'+str(ii)+str(jj)+str(kk)+'.npy'
            np.save(indexnpy,index)
            t2=time.time()
            print 'time used:',t2-t1
