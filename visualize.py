from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np



def cluster_visualize(km,data,txt):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    cluster_labels = km.predict(data)
    labels = pd.DataFrame(cluster_labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(pca_result[:,0],pca_result[:,1], c=labels[0],s=40,alpha=.5)
    plt.colorbar(scatter)
    plt.savefig(txt)


def make_graph(run,walk,jog,box,wave,clap,title,txt):
    N = 70
    ind = np.arange(0,350,5)
    print(ind)
    width = .5
    
    fig,ax = plt.subplots(1,1,figsize=(175,100))
   
    rects1 = ax.bar(ind,run,width,color='r')
    rects2 = ax.bar(ind+width,walk,width,color='g')
    rects3 = ax.bar(ind+2*width,jog,width,color='b')
    rects4 = ax.bar(ind+3*width,box,width,color='y')
    rects5 = ax.bar(ind+4*width,wave,width,color='c')
    rects6 = ax.bar(ind+5*width,clap,width,color='m')
    
    plt.legend((rects1[0],rects2[0],rects3[0],rects4[0],rects5[0],rects6[0]) ,("running","walking","joging","boxing","waving","claping"),prop={'size': 125})
    #ax.set_xticks(ind + width/6)
    #ax.set_xticklabels(np.arange(70))
    plt.xlabel('codewords',fontsize=150)
    plt.ylabel('average frequency count',fontsize=150)
    plt.suptitle(title,fontsize=250)
    
    #plt.xticks(np.arange(0, 70, step=1))
    plt.tick_params(axis='y', which='major', labelsize=100)
    plt.tick_params(axis='y', which='minor', labelsize=100)
    plt.tick_params(axis='x', which='major', labelsize=75)
    plt.tick_params(axis='x', which='minor', labelsize=75)
    
    '''y_pos = np.arange(len(val_m))
    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
    fig.suptitle(title,fontsize=15)
    ax1.bar(y_pos,val_m,align='center',alpha=1)
    ax1.set_title("magitude")
    ax1.set_ylim([0,80])
    y_pos = np.arange(len(val_a))
    ax2.bar(y_pos,val_a,align='center',alpha=1)
    ax2.set_title("angle")
    ax2.set_ylim([0,30])'''
    plt.savefig(txt)

def bar_graph(df_M,df_A,list_names,c):
    walk_m = np.empty(c, dtype=float)
    run_m = np.empty(c, dtype=float)
    box_m = np.empty(c, dtype=float)
    jog_m = np.empty(c, dtype=float)
    clap_m = np.empty(c, dtype=float)
    wave_m = np.empty(c, dtype=float)
    
    walk_a = np.empty(c, dtype=float)
    run_a = np.empty(c, dtype=float)
    box_a = np.empty(c, dtype=float)
    jog_a = np.empty(c, dtype=float)
    clap_a = np.empty(c, dtype=float)
    wave_a = np.empty(c, dtype=float)
    
    w=0
    r=0
    b=0
    j=0
    c=0
    wv=0
    
    for i in range(len(list_names)):
        if "boxing" in list_names[i]:
            b=b+1
            box_m = box_m + np.array(df_M.iloc[i])
            box_a = box_a + np.array(df_A.iloc[i])
        if "handclapping" in list_names[i]:
            c=c+1
            clap_m = clap_m + np.array(df_M.iloc[i])
            clap_a = clap_a + np.array(df_A.iloc[i])
        if "handwaving" in list_names[i]:
            wv=wv+1
            wave_m = wave_m + np.array(df_M.iloc[i])
            wave_a = wave_a + np.array(df_A.iloc[i])
        if "jogging" in list_names[i]:
            j=j+1
            jog_m = jog_m + np.array(df_M.iloc[i])
            jog_a = jog_a + np.array(df_A.iloc[i])
        if "running" in list_names[i]:
            r=r+1
            run_m = run_m + np.array(df_M.iloc[i])
            run_a = run_a + np.array(df_A.iloc[i])
        if "walking" in list_names[i]:
            w=w+1
            walk_m = walk_m + np.array(df_M.iloc[i])
            walk_a = walk_a + np.array(df_A.iloc[i])

    walk_a = walk_a/w
    walk_m = walk_m/w
    run_a = run_a/r
    run_m = run_m/r
    jog_a = jog_a/j
    jog_m = jog_m/j
    wave_a = wave_a/wv
    wave_m = wave_m/wv
    clap_a = clap_a/c
    clap_m = clap_m/c
    box_a = box_a/b
    box_m = box_m/b

    make_graph(run_m,walk_m,jog_m,box_m,wave_m,clap_m,"Magnitudes","mag.png")
    make_graph(run_a,walk_a,jog_a,box_a,wave_a,clap_a,"Angles","ang.png")








