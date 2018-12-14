import numpy as np
rs_dirs=(24,19,14,9,4,0)
fmb2_dirs = ('073','068','063','058','050','039')

dirname_rs = '/physics2/yingzhac/rockstar/'
treedir='/physics2/yingzhac/paper1/tree/z=3/'
matchdir='/physics2/yingzhac/summer2018/para_matching/'


logmbound=9.5
np.save(str(logmbound)+'.npy',logmbound)
eta=np.zeros(5)

ss=0
mbin=10
treeid=np.load(treedir+'treeid_'+str(mbin)+'_5.npy')
lines=len(treeid)
row_rs=np.zeros((lines,6))
row_sub=np.zeros((lines,6))
find=np.ones(lines,dtype = bool)


for i in range(6):
    np.save('file'+str(i)+'.npy',i)
    out=np.load(dirname_rs+'out_'+str(rs_dirs[i])+'.npy')
    matchid=np.load(matchdir+'sgmatch_ids_rs_to_mb2_'+fmb2_dirs[i]+'_'+str(logmbound)+'.npy')
    matchflag=np.load(matchdir+'sgmatch_flag_'+fmb2_dirs[i]+'_'+str(logmbound)+'.npy')
    for k in range(lines):
        temp=np.argwhere(out[:,0]==treeid[k,i])
        try:
            j=temp[0,0]
        except IndexError:
            find[k]=False
        else:
            row_rs[k,i]=j
            if matchflag[j]==1:
                sub_row=int(matchid[j])
                row_sub[k,i]=sub_row
            else:
                find[k]=False

row_rs=row_rs[find]
row_sub=row_sub[find]
eta[ss]=(len(row_sub)+0.0)/(lines+0.0)
np.save(treedir+'row_rs_10_5.npy',row_rs)
np.save(treedir+'row_sub_10_5.npy',row_sub)
    
savedir=treedir

np.save(savedir+'eta_10_5.npy',eta)




