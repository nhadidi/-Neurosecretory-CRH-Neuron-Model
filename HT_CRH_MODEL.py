'''Hypothalamic parvocellular neurosecretory CRH neuron model by Nima R. Hadidi'''


from brian2 import *
import numpy as np
start_scope()
defaultclock.dt = .01*ms

PVN_neurons = 29 # Number of neurons in modeled population
secrun=0.0  # Duration of simulated activity preceding recording of variables (seconds) 
recrun=0.1  # Duration of simulated activity during which variables are recorded (seconds)
IFF=0   # Factor by which mean number of inhibitory inputs is multiplied (66*IFF somatic and 66*IFF dendritic inhibitory inputs per second) 
EFF=0   # Factor by which mean number of excitatory inputs is multiplied (96*EFF somatic and 48*EFF dendritic excitatory inputs per second)

sroid =.0 # sets the strength of phasic inhibitory modulation (0, 1, or 2)
gabatonic =.0 # sets the strength of tonic inhibitory modulation (0, 1 or 2)
area_neuron=1200*um**2 # surface area of neuron
test=area_neuron

Ip0=0*nA # applied somatic current
gLs=0.02016 *msiemens*cm**-2*test # somatic leak conductance
gLd=0.02016 *msiemens*cm**-2*test # dendritic leak conductance
gNa=25.6  *msiemens*cm**-2*test # maximum sodium conductance
gKdr=3.6  *msiemens*cm**-2*test # maximum potassium delayed rectifier conductance
gCa=2.0  *msiemens*cm**-2*test # maximum calcium conductance
gKahp=0.06*msiemens*cm**-2*test # maximum SK potassium conductance (voltage and calcium dependent)
gKC=17  *msiemens*cm**-2*test # maximum BK potassium conductance (voltage and calcium dependent)
gm = 0.185*msiemens*cm**-2*test # maximum M-type potassium conductance
gKA=4000*msiemens*cm**-2*test # maximum conductance for A-type potassium conductance
T=34 # tau for A-type potassium conductance
ia=10**((T-24)/10) # A-type parameter
vhalfa=-21.5*mV # half activation voltage for A-type potassium current
vhalfi=-65*mV # half inactivation voltage for A-type potassium current
zi=3 # A-type inactivation parameter


VNa=55*mV #sodium equilibrium potential
VCa=120*mV # calcium equilibrium potential
VK=-90*mV # potassium equilibrium potential
VCl=-80*mV # GABA / chloride equilibrium potential
VL=-63*mV # leak equilibrium potential
Vsyn=0*mV # AMPA equilibrium potential
gc=5*msiemens*cm**-2*test # coupling conductance of somatic to dendritic compartment
pp=((396.08/(533.31+396.08))) # proportionality constant of cell body surface area to total surface area
Cm=1.5  *uF*cm**-2*test # cell capacitance
gbarsyn=.03*nS # single channel synaptic conductance
tausyn= 1.6*ms # AMPA conductance decay constant
tauNMDA= 50*ms # NMDA conductance decay constant
gton=0*(gabatonic==0)*ms/ms+8.3*(gabatonic==1)*ms/ms+20.3*(gabatonic==2)*ms/ms # sets the tonic conductance
tauisynslow=10.15*ms*(sroid==0)+12.6/9.6*10.15*ms*(sroid==1)+17.9/10.7*10.15*ms*(sroid==2) # sets the slow GABA decay constant
tauisynfast=2.0*ms*(sroid==0)+2.4/2.2*2*ms*(sroid==1)+2.2/1.8*2*ms*(sroid==2) # sets the fast GABA decay constant
tauweight=0*(sroid==0)+0.06*(sroid==1)+0.14*(sroid==2) # sets relative proportion of slow and fast components of GABAergic currents
scaler=(10**8*um**2/(test)) # scales dendritic calcium current

gKir=.02*msiemens*cm**-2*test # maximum inward rectifier current
amin=0.05 # minimum of inward rectifier activation
amax=0.8 # maximum inward rectifier activation
vhalfir=-90*mV # voltage of half inward rectifier activation

k=2 # inward rectifier activation parameter

Neuron_Eqs = '''
dVs/dt=(-gLs*(Vs-VL)-gNa*Minfs**2*hs*(Vs-VNa)-gKdr*(ns)*(Vs-VK)+(gc/pp)*(Vd-Vs)+(-Ip0-(Issyns+Iinssyns+INMDAs))-Im)/Cm : volt
dVd/dt=(-gLd*(Vd-VL)-ICad-gKahp*qd*(Vd-VK)-gKC*cd*chid*(Vd-VK)+(gc*(Vs-Vd))/(1.0-pp)-(Issynd+Iinssynd+INMDAd)-IKA-IKIR)/Cm : volt
dCad/dt=  -0.13*ICad/uamp/ms*scaler-0.075*Cad/ms : 1
dhs/dt=  alphahs-(alphahs+betahs)*hs : 1
dns/dt=  alphans-(alphans+betans)*ns : 1
dsd/dt=  alphasd-(alphasd+betasd)*sd : 1
dcd/dt=  alphacd-(alphacd+betacd)*cd : 1
dqd/dt=  alphaqd-(alphaqd+betaqd)*qd : 1
du/dt = alphau-(alphau+betau)*u : 1
alphau = 0.016/(exp(( Vs/mV + 45)/-23))/ms : Hz
betau = 0.016/(exp(( Vs/mV + 45)/18.8))/ms : Hz
ICad=     gCa*sd*sd*(Vd-VCa) : amp
alphams=  0.4*(-26-Vs/mV)/(exp((-26-Vs/mV)/4.0)-1.000001)/ms : Hz
betams=   0.28*(Vs/mV+14.9)/(exp((Vs/mV+14.9)/5.0)-1.000001)/ms : Hz
Minfs=    alphams/(alphams+betams) : 1 (constant over dt)
alphans=  0.016*(-10-Vs/mV)/(exp((-10-Vs/mV)/5.0)-1.000001)/ms : Hz
betans=   0.25*exp(-1.0-0.025*(Vs/mV-15))/ms : Hz
alphahs=  0.128*exp((-39.0-Vs/mV)/18.0)/ms : Hz
betahs=   4.0/(1.00001+exp((-16.0-Vs/mV)/5.0))/ms : Hz
alphasd=  1.6/(1.0001+exp(-0.072*(Vd/mV-5.0)))/ms : Hz
betasd=   0.02*(Vd/mV+8.9)/(exp((Vd/mV+8.9)/5.0)-1.0)/ms : Hz
alphacd=((Vd/mV<=-10)*exp((Vd/mV+50.0)/11-(Vd/mV+43.5)/27)/18.975+(Vd/mV>-10)*2.0*exp((-43.5-Vd/mV)/27.0))/ms  : Hz
betacd=   ((Vd/mV<=-10)*(2.0*exp((-43.5-Vd/mV)/27.0))-alphacd*ms)/ms : Hz
alphaqd=  clip(0.00002*Cad,0,0.01)/ms : Hz
betaqd=   0.001/ms : Hz
chid=     clip(Cad/250.0,0,1.0) : 1
aKir=amin+(amax-amin)/(1+exp((Vd-vhalfir)/mV/k)) : 1
IKIR=gKir*aKir*(Vd-VK) : amp
Im = gm*u**4*(Vs-VK) : amp
IKA=gKA*ja*ka*(Vd-VK) : amp
za=-1.8-(1/(1+exp(Vd/mV+40/ia))) : 1
dja/dt=(jainf-ja)/tauja : 1
dka/dt=(kainf-ka)/tauka : 1
alphaja=exp(289.44*za*(Vd-vhalfa)/mV/(8.315*(273.16+T))) : 1
betaja=exp(112.88*za*(Vd-vhalfa)/mV/(8.315*(273.16+T))) : 1
alphaka=exp(289.44*zi*(Vd-vhalfi)/mV/(8.315*(273.16+T))) : 1
betaka=exp(112.88*zi*(Vd-vhalfi)/mV/(8.315*(273.16+T))) : 1
jainf=1/(1+alphaja) : 1
tauja=clip(betaja/(ia*0.1*(1+alphaja)),0.2,inf)*ms : second
kainf=1/(1+alphaka) : 1
tauka=clip(0.26*(Vd/mV+50),2,inf)*ms : second
Issyns=gbarsyn*clip(ssyns,0,12000)*(Vs-Vsyn): amp
Issynd=gbarsyn*clip(ssynd,0,12000)*(Vd-Vsyn): amp
INMDAd=gbarsyn*clip(Sid,0,1400)*(1+0.28*exp(-0.062*(Vd/mV)))**(-1)*(Vd-Vsyn) : amp
INMDAs=gbarsyn*clip(Sis,0,1400)*(1+0.28*exp(-0.062*(Vs/mV)))**(-1)*(Vs-Vsyn) : amp
dssynd/dt=ssynde*22/(.08*ms)-ssynd/tausyn+abs(ssynd)**.5*randn()/.1/ms: 1
dssyns/dt=ssynse*22/(.08*ms)-ssyns/tausyn+abs(ssyns)**.5*randn()/.1/ms: 1
dSid/dt=ssynde*(1-Sid/150)*8/(.08*ms)-Sid/tauNMDA+abs(Sid)**.5*randn()/.4/ms: 1
dSis/dt=ssynse*(1-Sis/150)*8/(.08*ms)-Sis/tauNMDA+abs(Sis)**.5*randn()/.4/ms: 1
dinssynsslow/dt=(.5+tauweight)*ssynsi*57/(.08*ms)-inssynsslow/(tauisynslow) + abs(inssynsslow)**.5*randn()/.1/ms: 1
dinssyndslow/dt=(.5+tauweight)*ssyndi*57/(.08*ms)-inssyndslow/(tauisynslow) + abs(inssyndslow)**.5*randn()/.1/ms: 1
dinssynsfast/dt=(.5-tauweight)*ssynsi*57/(.08*ms)-inssynsfast/(tauisynfast) + abs(inssynsfast)**.5*randn()/.05/ms: 1 
dinssyndfast/dt=(.5-tauweight)*ssyndi*57/(.08*ms)-inssyndfast/(tauisynfast) + abs(inssyndfast)**.5*randn()/.05/ms: 1
dgtonics/dt=abs(gton)**.5*randn()/.1/ms-gtonics/(1*ms): 1
dgtonicd/dt=abs(gton)**.5*randn()/.1/ms-gtonicd/(1*ms): 1
inssyns=abs(inssynsslow+inssynsfast+gton+gtonics): 1
inssynd=abs(inssyndslow+inssyndfast+gton+gtonicd): 1
dssynde/dt=-ssynde/(0.08*ms) : 1
dssynse/dt=-ssynse/(0.08*ms) : 1
dssyndi/dt=-ssyndi/(0.08*ms) : 1
dssynsi/dt=-ssynsi/(0.08*ms) : 1
Iinssyns=gbarsyn*clip(inssyns,0,12000)*(Vs-VCl): amp
Iinssynd=gbarsyn*clip(inssynd,0,12000)*(Vd-VCl): amp
counter:1
dVCl/dt=0*mV/ms : volt
'''


PVN_neurongroup = NeuronGroup(PVN_neurons, Neuron_Eqs, threshold='Vs>-0*mV',refractory='Vs>-35*mV', method='rk4')


Excited_neurongroup = PoissonInput(PVN_neurongroup,'ssynde', 40, EFF*2.4*Hz,'1+randn()*.3')
Excites_neurongroup = PoissonInput(PVN_neurongroup, 'ssynse', 20, EFF*2.4*Hz, '1+randn()*.3')




Inhibd_neurongroup = PoissonInput(PVN_neurongroup,'ssyndi', 30, IFF*2.2*Hz,'1+randn()*.3')
Inhibs_neurongroup = PoissonInput(PVN_neurongroup, 'ssynsi', 30, IFF*2.2*Hz, '1+randn()*.3')




M = StateMonitor(PVN_neurongroup, 'Vs', record=True)
N = SpikeMonitor(PVN_neurongroup)




PVN_neurongroup.Vs='(randn()*60*.05-60)*mV'
PVN_neurongroup.Vd='(randn()*65*.05-65)*mV'
PVN_neurongroup.hs=.999
PVN_neurongroup.ns=.0001
PVN_neurongroup.sd=.009
PVN_neurongroup.cd=.007
PVN_neurongroup.qd=.00025
PVN_neurongroup.Cad=.20
PVN_neurongroup.VCl=-80*mV





for x in range(29):
    PVN_neurongroup[x].Ip0=(20)*pA
run(secrun*1000*ms, report='text')
for x in range(29):
    PVN_neurongroup[x].Ip0=(40-5*x)*pA
run(recrun*1000*ms, report='text')
for x in range(29):
    PVN_neurongroup[x].Ip0=(0)*pA
run(secrun*1000*ms, report='text')


for x in range(0,23,2):
    plot(M.t/ms,M[x].Vs/mV)

for x in range(0,23,2):
    plot(M.t[::10]/ms,M[x].Vs[::10]/mV)
    
    
freqlist=zeros(29)
for x in N.i:
    freqlist[x]+=1
plot(np.linspace(-40,70,12),freqlist[0:23:2])
