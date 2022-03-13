import math
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

# initialize
numPN  = 200           # pyramidal neurons
numIN  = 50            # interneurons
tcells = (numPN + numIN)  # total cells

# parameters
trials = 1
ETIME  = 2000                 # length of simulation (ms)
TSTEP  = 0.1                  # time step of computation
NSTEPS = int((ETIME/TSTEP)+1) # number of computational steps 
ASIZE  = int(NSTEPS+1)        # array size variable

g_L     = 0.05                # leak conductance
V_rest  = 0                   # resting potential
E_AMPA  = 4.67                # AMPAR reversal potential
E_GABA  = -0.67               # GABAR reversal potential
E_NMDA  = 4.67
tau_m   = 20                  # time constant (ms)
tau_ref = 2                   # refractory period (ms)
refsteps  = tau_ref/TSTEP     # refractory period in time steps
threshold = 1                 # spike and then reset at V_m = 1
maxspikes = 500

tau_A = 3.5
tau_B = 436.9
tp = (tau_A*tau_B)/(tau_B - tau_A) * np.log(tau_B/tau_A)
factor = -math.exp(-tp/tau_A) + math.exp(-tp/tau_B)

# NMDA act/deact
PNfac = 0
INfac = 0

# spike probs
ININP = 0.45
INPNP = 0.45
PNINP = 0.35
PNPNP = 0.3

# amplitudes
ININA = 0.65
INPNA = 0.65
PNINA = 0.2
PNPNA = 0.02

# synaptic delay
ININSD = 1.1
INPNSD = 0.6
PNPNSD = 1.0
PNINSD = 2.3

# decay rates
ININD = 1.2
INPND = 2.3 #3.3
PNPND = 1.7 #1.7
PNIND = 1.6

# initialize storage arrays
coupling  = np.zeros((tcells,tcells),dtype=int)     # array of T/F of synapses
time      = np.zeros((ASIZE,))

INvolt  = np.zeros((numIN,ASIZE))
INexc   = np.zeros((numIN,ASIZE))
INinh   = np.zeros((numIN,ASIZE))         
PNvolt  = np.zeros((numPN,ASIZE))
PNinh   = np.zeros((numPN,ASIZE))
PNexc   = np.zeros((numPN,ASIZE))
INac = np.zeros((numIN,ASIZE))
INde = np.zeros((numIN,ASIZE))
PNac = np.zeros((numPN,ASIZE))
PNde = np.zeros((numPN,ASIZE))

# temporary storage variables (V, exc, inh) 
INv   = np.zeros((numIN,2))
INe   = np.zeros((numIN,2))
INi   = np.zeros((numIN,2))
PNv   = np.zeros((numPN,2))
PNi   = np.zeros((numPN,2))
PNe   = np.zeros((numPN,2))
PNA   = np.zeros((numPN,2))
PNB   = np.zeros((numPN,2))
INA   = np.zeros((numIN,2))
INB   = np.zeros((numIN,2))

# track refractory periods for each neuron
INref = np.zeros((numIN,))
PNref = np.zeros((numPN,))

# track of spike counts of each neuron (all trials)
INspikecount  = np.zeros((trials,numIN),dtype=int)
PNspikecount  = np.zeros((trials,numPN),dtype=int)

# keep track of how many spikes have been inputed
INspiketrack  = np.zeros((numIN,),dtype=int)
INspiketrack2 = np.zeros((numIN,),dtype=int) 
PNspiketrack  = np.zeros((numPN,),dtype=int) 
PNspiketrack2 = np.zeros((numPN,),dtype=int) 

# track spike times of each neuron (all trials)
INspiketimes  = np.full((trials,numIN,maxspikes),-1,dtype=float)
PNspiketimes  = np.full((trials,numPN,maxspikes),-1,dtype=float)  

# 
tempexc = np.zeros((ASIZE,))
tempinh = np.zeros((ASIZE,))
tempsyn = np.zeros((ASIZE,))

IN_NMDA = np.zeros((numIN,2))
PN_NMDA = np.zeros((numPN,2))
INNMDA = np.zeros((numIN,ASIZE))
PNNMDA = np.zeros((numPN,ASIZE))

##############################################################################################

def SetCoupling(coupling,seed):
    np.random.seed(seed)
    
    for i in range(1,numPN+1):
        for j in range(numPN+1,numPN+numIN+1):           # PN -> IN
            coupling[i-1,j-1] = np.random.uniform(0,1) < PNINP
        for j in range(1,numPN+1):                       # PN -> PN
            if i == j:
                coupling[i-1,j-1] = 0
            else: 
                coupling[i-1,j-1] = np.random.uniform(0,1) < PNPNP
    
    for i in range(numPN+1,numPN+numIN+1):
        for j in range(1,numPN+1):                       # IN -> PN
            coupling[i-1,j-1] = np.random.uniform(0,1) < INPNP
        for j in range(numPN+1,numPN+numIN+1):           # IN -> IN
            if i == j: 
                coupling[i-1,j-1] = 0
            else: 
                coupling[i-1,j-1] = np.random.uniform(0,1) < ININP

def noisePN(): 
    mean = 0.08
    std = 0.4
    noise = np.random.normal(mean, std)
    return(noise)

def noiseIN(): 
    mean = 0.0
    std = 0.2
    noise = np.random.normal(mean, std)
    return(noise)


##############################################################################################

SetCoupling(coupling,22)
np.random.seed()

INspiketimes  = np.full((trials,numIN,maxspikes),-1,dtype=float)
PNspiketimes  = np.full((trials,numPN,maxspikes),-1,dtype=float)  

for trial in range(1, trials+1):
    j = 0
    temp = 0

    for i in range(1,numIN+1):                            # reset current counters
        INspikecount[trial-1,i-1]=0
        INspiketrack[i-1]=0
        INspiketrack2[i-1]=0
        INv[i-1,1]=0
        INe[i-1,1]=0
        INi[i-1,1]=0
        INref[i-1]=0
        INB[i-1,1]=0
        INA[i-1,1]=0

    for i in range(1,numPN+1):
        PNspikecount[trial-1,i-1]=0
        PNspiketrack[i-1]=0
        PNspiketrack2[i-1]=0
        PNv[i-1,1]=0
        PNe[i-1,1]=0
        PNi[i-1,1]=0
        PNref[i-1]=0
        PNB[i-1,1]=0
        PNA[i-1,1]=0

    for i in range(1,NSTEPS+1): 

        for k in range(1, numIN+1):                 # move current time slots into previous time slots
            INv[k-1,0] = INv[k-1,1]                 # later use time slots to check if a cell spiked
            INe[k-1,0] = INe[k-1,1]
            INi[k-1,0] = INi[k-1,1]
            INB[k-1,0] = INB[k-1,1]
            INA[k-1,0] = INA[k-1,1]
        for k in range(1, numPN+1):
            PNv[k-1,0] = PNv[k-1,1]
            PNi[k-1,0] = PNi[k-1,1]
            PNe[k-1,0] = PNe[k-1,1]
            PNA[k-1,0] = PNA[k-1,1]
            PNB[k-1,0] = PNB[k-1,1]
            
        # refractory period
        for k in range(1,numIN+1):                  # track IN refractory period
            if INref[k-1] >= refsteps: 
                INref[k-1] = 0
            if INref[k-1] != 0: 
                INref[k-1] += 1
        for k in range(1,numPN+1):                   # track PN refractory period
            if PNref[k-1] >= refsteps: 
                PNref[k-1] = 0
            if PNref[k-1] != 0: 
                PNref[k-1] += 1

        # V_m
        for k in range(1,numIN+1): 
            if INref[k-1] != 0:
                INv[k-1,1] = V_rest
            else: 
                Mg = 1 / (8 + math.exp(-8 * (INv[k-1,0] - 0.6)))
                I_NMDA = -INfac*(INB[k-1,0]-INA[k-1,0])*Mg*(INv[k-1,0]-E_NMDA)
                dvdt = (-g_L*(INv[k-1,0]-V_rest) - INe[k-1,0]*(INv[k-1,0]-E_AMPA) 
                        - INi[k-1,0]*(INv[k-1,0]-E_GABA) + noiseIN() + I_NMDA)
                INv[k-1,1] = INv[k-1,0] + TSTEP*dvdt
            dAdt = -INA[k-1,0]/tau_A
            dBdt = -INB[k-1,0]/tau_B
            IN_NMDA[k-1,1] = I_NMDA
            INA[k-1,1] = INA[k-1,0] + TSTEP*dAdt
            INB[k-1,1] = INB[k-1,0] + TSTEP*dBdt
            INi[k-1,1] = INi[k-1,0]*(math.exp((1/(-ININD))*TSTEP))
            INe[k-1,1] = INe[k-1,0]*(math.exp((1/(-PNIND))*TSTEP))
        
        for k in range(1,numPN+1): 
            if PNref[k-1] != 0: 
                PNv[k-1,1] = V_rest
            else: 
                Mg = 1 / (8 + math.exp(-8 * (PNv[k-1,0] - 0.6)))
                I_NMDA = -PNfac*(PNB[k-1,0]-PNA[k-1,0])*Mg*(PNv[k-1,0]-E_NMDA)
                
                dvdt = (-g_L*(PNv[k-1,0]-V_rest) - PNe[k-1,0]*(PNv[k-1,0]-E_AMPA) 
                        - PNi[k-1,0]*(PNv[k-1,0]-E_GABA) + noisePN() + I_NMDA)
                PNv[k-1,1] = PNv[k-1,0] + TSTEP*dvdt
            
            PN_NMDA[k-1,1] = I_NMDA
            dAdt = -PNA[k-1,0]/tau_A
            dBdt = -PNB[k-1,0]/tau_B
            PNA[k-1,1] = PNA[k-1,0] + TSTEP*dAdt
            PNB[k-1,1] = PNB[k-1,0] + TSTEP*dBdt
            PNi[k-1,1] = PNi[k-1,0]*(math.exp((1/(-INPND))*TSTEP))
            PNe[k-1,1] = PNe[k-1,0]*(math.exp((1/(-PNPND))*TSTEP))

        for k in range(1,numIN+1):                  # check and track IN spikes
            if (INv[k-1,1] >= threshold and INv[k-1,0] < threshold):
                INv[k-1,1] = V_rest
                INspikecount[trial-1,k-1] += 1
                INspiketimes[trial-1,k-1,INspikecount[trial-1,k-1]-1] = i*TSTEP
                INref[k-1] = 1
        for k in range(1,numPN+1):                  # check and track PN spikes
            if (PNv[k-1,1] >= threshold and PNv[k-1,0] < threshold):
                PNv[k-1,1] = V_rest
                PNspikecount[trial-1,k-1] += 1
                PNspiketimes[trial-1,k-1,PNspikecount[trial-1,k-1]-1] = i*TSTEP
                PNref[k-1] = 1

        for k in range(1,numIN+1):                  # ININ inputs
            temp=INspiketrack2[k-1]+1
            for m in range(temp,INspikecount[trial-1,k-1]+1):
                if (INspiketimes[trial-1,k-1,m-1] + ININSD <= i*TSTEP and 
                    INspiketimes[trial-1,k-1,m-1] + ININSD > (i-1)*TSTEP): 
                    INspiketrack2[k-1] += 1
                    for n in range(1,numIN+1): 
                        INi[n-1,1] += ININA * coupling[numPN+k-1,numPN+n-1]
            temp=INspiketrack[k-1]+1                # INPN inputs
            for m in range(temp,INspikecount[trial-1,k-1]+1):
                if (INspiketimes[trial-1,k-1,m-1] + INPNSD <= i*TSTEP and 
                    INspiketimes[trial-1,k-1,m-1] + INPNSD > (i-1)*TSTEP):
                    INspiketrack[k-1] += 1
                    for n in range(1,numPN+1):
                        PNi[n-1,1] += INPNA * coupling[numPN+k-1,n-1]

        for k in range(1,numPN+1):                  # PNPN inputs
            temp=PNspiketrack[k-1]+1
            for m in range(temp,PNspikecount[trial-1,k-1]+1):
                if (((PNspiketimes[trial-1,k-1,m-1] + PNPNSD) <= i*TSTEP) and
                   ((PNspiketimes[trial-1,k-1,m-1] + PNPNSD) > (i-1)*TSTEP)): 
                    PNspiketrack[k-1] += 1
                    for n in range(1,numPN+1): 
                        PNe[n-1,1] += PNPNA * coupling[k-1,n-1]
                        PNA[n-1,1] += PNAwf * 1 * coupling[k-1,n-1] * factor
                        PNB[n-1,1] += PNBwf * 1 * coupling[k-1,n-1] * factor
            temp=PNspiketrack2[k-1]+1               # PNIN inputs
            for m in range(temp,PNspikecount[trial-1,k-1]+1):
                if (((PNspiketimes[trial-1,k-1,m-1] + PNINSD) <= i*TSTEP) and
                   ((PNspiketimes[trial-1,k-1,m-1] + PNINSD) > (i-1)*TSTEP)): 
                    PNspiketrack2[k-1] += 1
                    for n in range(1,numIN+1):
                        INe[n-1,1] += PNINA * coupling[k-1,numPN+n-1]
                        INA[n-1,1] += INAwf * 1 * coupling[k-1,numPN+n-1] * factor
                        INB[n-1,1] += INBwf * 1 * coupling[k-1,numPN+n-1] * factor

        # record everything
        j += 1
        time[j-1] = i*TSTEP
        for k in range(1,numIN+1): 
            INvolt[k-1,j-1]  = INv[k-1,1]
            INinh[k-1,j-1]   = INi[k-1,1]
            INexc[k-1,j-1]   = INe[k-1,1]
            INac[k-1,j-1] = INA[k-1,1]
            INde[k-1,j-1] = INB[k-1,1]
            INNMDA[k-1,j-1] = IN_NMDA[k-1,1]
        for k in range(1,numPN+1): 
            PNvolt[k-1,j-1]  = PNv[k-1,1]
            PNinh[k-1,j-1]   = PNi[k-1,1]
            PNexc[k-1,j-1]   = PNe[k-1,1]
            PNac[k-1,j-1] = PNA[k-1,1]
            PNde[k-1,j-1] = PNB[k-1,1]
            PNNMDA[k-1,j-1] = PN_NMDA[k-1,1]