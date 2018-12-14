import numpy as np

sf=(0,5,10,15,20,24)
treedir='/physics2/yingzhac/paper1/tree/z=3/'
treedir2='/physics2/yingzhac/paper1/tree/z=3/'


treeid=np.zeros((300000,6))
treemass=np.zeros((300000,6))
treepos=np.zeros((300000,6,3))


m2=10**17


logm=10
m1=5*10**logm
s=0
print logm
for i in range(4):
    for j in range(4):
        for k in range(4):
            print 'Now in tree: ',i,j,k
            treename='tree'+str(i)+str(j)+str(k)+'.npy'
            indicename='index'+str(i)+str(j)+str(k)+'.npy'
            tree=np.load(treedir+treename)
            indice=np.load(treedir+indicename)
            n=indice.shape[0]
            for line in range(n):
                row_des=int(indice[line,0])
                row_anc=int(indice[line,24])
                if m1<tree[row_des,10]<m2 and row_anc>=0:
                    for stage in range(6):
                        row=int(indice[line,sf[stage]])
                        treeid[s,stage]=tree[row,30]
                        treemass[s,stage]=tree[row,10]
                        treepos[s,stage,0]=tree[row,17]*1000
                        treepos[s,stage,1]=tree[row,18]*1000
                        treepos[s,stage,2]=tree[row,19]*1000
                    s+=1
print 'total:',s
treeid=treeid[:s]
treemass=treemass[:s]
treepos=treepos[:s]


np.save(treedir2+'treeid_'+str(logm)+'_5.npy',treeid)
np.save(treedir2+'treemass_'+str(logm)+'_5.npy',treemass)
np.save(treedir2+'treepos_'+str(logm)+'_5.npy',treepos)













