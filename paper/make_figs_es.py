import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D="../results/"
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
                     "xtick.major.width":0.6,"ytick.major.width":0.6,"legend.frameon":False,"savefig.bbox":"tight","savefig.pad_inches":0.02})
C={"net":"#1f4e79","rf":"#c05a1c","ctl":"#7a7a7a","tcn":"#3d8f6b","base":"#b9b9b9"}
W=3.45
# Fig 1: example signals
d=np.load(D+"dataset_A.npz"); X=d["X"]; names=list(d["whisker_names"]); y=d["y"]
m=np.abs(X[...,:3]).max(axis=(1,2,3)); i=int(np.argsort(m)[int(0.7*len(m))])
pk=np.abs(X[i,:,:,2]).max(0); top=np.argsort(pk)[::-1][:4]; t=np.arange(X.shape[1])+5
fig,ax=plt.subplots(2,1,figsize=(W,2.6),sharex=True)
cols=["#1f4e79","#c05a1c","#3d8f6b","#8a5aa8"]
for k,w in enumerate(top):
    ax[0].plot(t,X[i,:,w,2],lw=0.8,color=cols[k],label=names[w]); ax[1].plot(t,X[i,:,w,4],lw=0.8,color=cols[k])
ax[0].set_ylabel(r"$M_z$ (unid. del simulador)"); ax[1].set_ylabel(r"$F_y$ (unid. del simulador)"); ax[1].set_xlabel("tiempo (ms)")
ax[0].legend(ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.32),handlelength=1.2,columnspacing=1.0)
fig.savefig("figs_es/fig_signals.pdf"); plt.close(fig); print("example sim", i, "peg", y[i], "whiskers", [names[w] for w in top])
# Fig 2: predicted vs true positions
p=np.load(D+"results_A_s2/preds.npz"); tr,pr=p["true"],p["cnn_lstm"]; e=np.sqrt(((tr-pr)**2).sum(1))
fig,ax=plt.subplots(figsize=(W,2.5))
for a,b in zip(tr,pr): ax.plot([a[0],b[0]],[a[1],b[1]],color="#9aa7b5",lw=0.5,zorder=1)
sc=ax.scatter(tr[:,0],tr[:,1],c=e,s=9,cmap="viridis",vmin=0,vmax=6,zorder=2,linewidths=0)
cb=fig.colorbar(sc,ax=ax,pad=0.02); cb.set_label("error de posición (mm)"); cb.outline.set_linewidth(0.5)
ax.set_xlabel("x (mm), rostral"); ax.set_ylabel("y (mm), lateral"); ax.set_aspect("equal")
fig.savefig("figs_es/fig_scatter.pdf"); plt.close(fig); print("scatter: median err %.2f, 90th %.2f, n=%d"%(np.median(e),np.percentile(e,90),len(e)))
# Fig 3: robustness to noise
r=json.load(open(D+"robust_A/robustness.json")); lv=[0,5,10,20]; keys=["clean","noise_0.05","noise_0.10","noise_0.20"]
fig,ax=plt.subplots(figsize=(W,2.1))
for k,lab,c,mk in (("cnn_lstm","CNN-LSTM",C["net"],"o"),("rf_contact_map","RF sobre mapa de contacto",C["rf"],"s")):
    mu=np.array([r[k][q][0] for q in keys]); sd=np.array([r[k][q][1] for q in keys])
    ax.errorbar(lv,mu,yerr=sd,color=c,marker=mk,ms=3.5,lw=1.1,capsize=2,label=lab)
ax.set_xticks(lv); ax.set_xlabel("ruido gaussiano aditivo (% de la desv. del canal)"); ax.set_ylabel("error de posición (mm)"); ax.set_ylim(0,6.8); ax.legend(loc="upper left")
fig.savefig("figs_es/fig_noise.pdf"); plt.close(fig)
# Fig 4: shape accuracy across three designs
S=[json.load(open(D+f"{n}/summary.json")) for n in ("results_B","results_B2","results_B3")]
mods=[("pose_only","Control solo pose",C["ctl"]),("rf_contact_map","RF sobre mapa de contacto",C["rf"]),("cnn_lstm","CNN-LSTM",C["net"]),("tcn","TCN",C["tcn"])]
fig,ax=plt.subplots(figsize=(W,2.5)); bw=0.19
for j,(k,lab,c) in enumerate(mods):
    mu=[s[k]["acc"][0] for s in S]; sd=[s[k]["acc"][1] for s in S]
    ax.bar(np.arange(3)+(j-1.5)*bw,mu,bw*0.92,yerr=sd,color=c,label=lab,error_kw=dict(lw=0.6,capsize=1.5))
ax.axhline(1/3,color="k",lw=0.7,ls="--"); ax.text(0.5,1/3-0.045,"azar",ha="center",fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(["D1\ncabeza quieta\nforma acoplada","D2\ncabeza quieta\ndesacoplada","D3\ncabeza avanza\ndesacoplada"],fontsize=6.5)
ax.set_ylabel("exactitud de forma"); ax.set_ylim(0,0.8); ax.legend(ncol=2,loc="lower center",bbox_to_anchor=(0.5,1.0),handlelength=1.0,columnspacing=1.2)
fig.savefig("figs_es/fig_shape.pdf"); plt.close(fig)
