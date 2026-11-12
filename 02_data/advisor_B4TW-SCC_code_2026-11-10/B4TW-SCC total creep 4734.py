# import functions from the libraries
import math
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
import itertools
import numpy as np
import pyodbc 
import getpass 
import pyodbc 
import sys
from pandas import DataFrame
import pandas as pd
from pandas.io.sql import read_sql
from decimal import *
import xlwings as xw
import xlwt
#MODEL Ｂ4TW2020 total creep
plot.figure(dpi=110)
# Parameters definition

# ktaua------------------------------------------------------------------------
def ktaua_cal(aggtype):
    if aggtype==1:
        return 0.06
    elif aggtype==2:
        return 0.59
    elif aggtype==3:
        return 1.8
    elif aggtype==4:
        return 2.7
    elif aggtype==5:
        return 4.0
    else:
        print ("Aggregate type error( must be 1~5 )")

# kea--------------------------------------------------------------------------
def kea_cal(aggtype):
    if aggtype==1:
        return 0.76
    elif aggtype==2:
        return 0.71
    elif aggtype==3:
        return 0.95
    elif aggtype==4:
        return 2.0
    elif aggtype==5:
        return 1.05
    else:
        print ("Aggregate type error( must be 1~5 )")
#Kh----------------------------------------------------------------------------
def Kh_cal(RH):
    if RH<=0.98:
        return 1-RH**3
    elif RH==1:
        return -0.2
    elif 0.98<RH<1:
        return 12.94*(1-RH)-0.2
    else:
        print ("Humidity error( must be 0~1 )")

#Ks----------------------------------------------------------------------------
def Ks_cal(geo):
    if geo==1:
        return 1
    elif geo==2:
        return 1.15
    elif geo==3:
        return 1.25
    elif geo==4:
        return 1.3
    elif geo==5:
        return 1.55
    else:
        print ("Geometry error( must be 1~5 )")

# =============================================================================

# Coarse aggregate
#Table1------------------------------------------------------------------------
def Table1_cal(cement):
    if cement==1:
        return [0.016, -0.30, -0.04, -0.05, 440e-6, -0.6,  1.10, 0.07, 0.35]
    elif cement==2:
        return [0.010, -0.30,  4.50,  2.50, 400e-6, -0.6,  3.20, 0.07, 0.80]
    elif cement==3:
        return [0.060, -0.30, -2.60, -2.00, 860e-6, -2.6,  0.40, 0.07, 0.46]
    else:
        print ("Cement type error( must be 1~3 )")
#Table3------------------------------------------------------------------------
#4734
def Table3_cal(cement):
    if cement==1:
        return [1.30, 50.0e-3, 40.0e-3, 4.0e-3,  700e-6, 7, 2.0, 1.1, 0.2, -0.8, 0.3, -0.7,  2.58, -0.76, -0.5, 4.00]
    elif cement==2:
        return [0.90, 30.0e-3, 40.0e-3, 3.4e-3,  300e-6, 7, 2.5, 1.1, 0.7, -0.8, 0.3, -0.7,  2.58, -0.86, -0.5, 1.00]
    elif cement==3:
        return [0.60, 17.4e-3, 39.9e-3, 3.4e-3, 94.6e-6, 1, 3.0, 1.0, 1.0, -1.1, 0.4, -0.9,  2.45, -0.85, -1.0, 0.78]
    else:
        print ("Cement type error( must be 1~3 )")


# Original values (B4TW-2020)
# =============================================================================
# def Table3_cal(cement):
#     if cement==1:
#         return [0.85, 45.0e-3, 40.0e-3, 3.4e-3,  700e-6, 7, 2.5, 1.1, 0.7, -0.8, 0.3, -0.7,  2.58, -0.86, -0.5, 1.00, 1.0]
#     elif cement==2:
#         return [0.90, 30.0e-3, 40.0e-3, 3.4e-3,  300e-6, 7, 2.5, 1.1, 0.7, -0.8, 0.3, -0.7,  2.58, -0.86, -0.5, 1.00, 1.0]
#     elif cement==3:
#         return [0.60, 17.4e-3, 39.9e-3, 3.4e-3, 94.6e-6, 1, 3.0, 1.0, 1.0, -1.1, 0.4, -0.9,  2.45, -0.85, -1.0, 0.78, 1.0]
#     else:
#         print ("Cement type error( must be 1~3 )")
# =============================================================================


#t-----------------------------------------------------------------------------
def BTh_cal(Tcur):
    return math.exp(4000*(1/293.-1./(Tcur+273)))
def BTs_cal(T):
    return math.exp(4000*(1/293.-1./(T+273)))
def BTc_cal(T):
    return math.exp(4000*(1/293.-1./(T+273)))
    
def t0wave_cal(t0,BTh):
    return t0*BTh
def tphat_cal(t0,tp,BTh,BTs):
    return t0*BTh+(tp-t0)*BTs

#fcmtp-------------------------------------------------------------------------
def fcmtp_cal(a,b,fc,tp):
    return (tp/(a+b*tp))*fc

#4734
#Etp---------------------------------------------------------------------------
def Ecmtp_cal(fcmtp,fc):
    return (fcmtp**0.5)*4734
#E28--s-------------------------------------------------------------------------
def E28_cal(fc):
        return (fc**0.5)*4734


#Original formulae
# =============================================================================
# #Etp---------------------------------------------------------------------------
# def Ecmtp_cal(tp,fcmtp,fc):
#     if tp<28:
#         return (fcmtp**0.5)*3831
#     else:
#         return (fc**0.5)*3831
# =============================================================================


#E(t)--------------------------------------------------------------------------
def E_cal(t):
    return E28*(t/(4+(6./7)*t))**0.5

#tau0--------------------------------------------------------------------------
def tau0_cal(acm,wcm,cm,taucem,ptaua,ptauw,ptauc):
    return taucem*((acm/3.4)**ptaua)*((wcm/0.36)**ptauw)*((4.6*cm/2350)**ptauc)
    
#taush-------------------------------------------------------------------------
def taush_cal(D,Ks,ktaua,tau0):
    return tau0*ktaua*((Ks*D)**2)
    
#e0----------------------------------------------------------------------------
#def e0_cal(acm,wcm,cm,ecem,pea,pew,pec,sa,pesa): # Sandrate
#    return ecem*((acm/4.5)**pea)*((wcm/0.3)**pew)*((4.7*cm/2350)**pec)*((sa/0.5)**pesa)
def e0_cal(acm,wcm,cm,ecem,pea,pew,pec,ca,peca): # Coarse aggregate
    return ecem*((acm/3.4)**pea)*((wcm/0.36)**pew)*((4.6*cm/2350)**pec)*((ca/840)**peca)

#esh8--------------------------------------------------------------------------
def esh8_cal(e0,kea,E607_m,Et0_taush_m,fshcem):
    return -1*e0*kea*E607_m/Et0_taush_m

#S(t)--------------------------------------------------------------------------
def S_cal(twave_array,taush,fshtau):
    return [math.tanh((n/(taush*fshtau))**0.5) for n in twave_array]
            
#esh---------------------------------------------------------------------------
def esh_cal(esh8,S,Kh,curing):
    if curing==1:
        return 1.22*S*Kh*esh8*fshcem
    if curing==3:
       return S*Kh*esh8*fshcem 
    
#q1~q5-------------------------------------------------------------------------
def q1_cal(p1,Ecmtp):
    return p1/Ecmtp

#def q2_cal(p2,wcm,p2w,cm,p2c,fp2,sa,p2sa): # Sandrate
#    return fp2*p2*((wcm/0.3)**p2w)*((4.7*cm/2350)**p2c)*((sa/0.5)**p2sa)/1000
def q2_cal(p2,wcm,p2w,cm,p2c,fp2,ca,p2ca): # Coarse aggregate
    return fp2*p2*((wcm/0.36)**p2w)*((4.6*cm/2350)**p2c)*((ca/840)**p2ca)/1000


def q3_cal(q2,p3,acm,wcm,p3a,p3w,fp3):
    return fp3*p3*q2*((acm/3.4)**p3a)*((wcm/0.36)**p3w)
def q4_cal(p4,acm,wcm,p4a,p4w,fp4):
    return fp4*p4*((acm/3.4)**p4a)*((wcm/0.36)**p4w)/1000
def q5_cal(p5,acm,wcm,p5a,p5w,Kh,esh8,p5e,fp5):
    return fp5*p5*((acm/3.4)**p5a)*((wcm/0.36)**p5w)*(math.fabs(Kh*esh8))**p5e/1000

#C0----------------------------------------------------------------------------
def C0_cal(q2,q3,q4,Q,that_array,tphat_array):
    return q2*Q+q3*np.array([math.log(1+n**0.1) for n in (that_array-tphat_array)])+q4*np.array([math.log(n) for n in (that_array/tphat_array)])

#Cd----------------------------------------------------------------------------
def r_cal(tphat):
    return 1.7*(tphat**0.12)+8
def Z_cal(tphat,tphat_array,that_array):
    return [math.log(1+n**0.1)*tphat**-0.5 for n in (that_array-tphat_array)]
def Qf_cal(tphat):
    return 1./(0.086*tphat**(2./9)+1.21*tphat**(4./9))
def Q_cal(Qf,r,Z):
    return Qf*(1+(Qf/Z)**r)**(-1/r)
def tp0hat_array_cal(tphat_array,t0wave_array):
    return np.maximum(tphat_array,t0wave_array)
def H_cal(that_array,t0wave_array,RH):
    return [1-(1-RH)*math.tanh((n/taush)**0.5) for n in (that_array-t0wave_array)]
def Hc_cal(tp0hat_array,t0wave_array,RH):
    return [1-(1-RH)*math.tanh((n/taush)**0.5) for n in (tp0hat_array-t0wave_array)]

# Coarse aggregate
def Table4S_cal(Slag,FlyAsh,SiO2,NP,Filler,wcm):#Slag<=70, FlyAsh<=70, SiO2<=15, NP<=70
    if Slag==0 and FlyAsh==0 and SiO2==0 and NP==0 and Filler>0:            
        return [1.20, 0.60, 1.00, 1.00]        
    elif Slag>0 and FlyAsh==0 and SiO2==0 and NP==0:            
        return [1.30, 0.80, 0.85, 1.00]        
    elif Slag==0 and FlyAsh>0 and SiO2==0 and NP==0:            
        return [1.05, 0.90, 1.15, 0.95]
    elif Slag>0 and FlyAsh>0 and SiO2==0 and NP==0:           
        return [1.00, 0.70, 1.10, 1.00]
    elif Slag==0 and FlyAsh==0 and SiO2>0 and NP==0:           
        return [1.00, 0.90, 1.00, 1.00]
    elif Slag>0 and FlyAsh==0 and SiO2>0 and NP==0:           
        return [1.50, 1.00, 0.90, 1.10]
    elif Slag==0 and FlyAsh>0 and SiO2>0 and NP==0:
        return [0.95, 0.65, 1.15, 0.85]
    elif Slag>0 and FlyAsh>0 and SiO2>0 and NP==0:            
        return [0.95, 0.60, 0.85, 1.15]
    elif NP>0:
        return [0.95, 0.70, 0.90, 1.00]
    else:
        return [1.00, 1.00, 1.00, 1.00]
        
#4734
def Table4C_cal(Slag,FlyAsh,SiO2,wcm,rtype):#Slag<=70, FlyAsh<=70, SiO2<=15
        if Slag>0 and FlyAsh==0 and SiO2==0:
            return [1.0,  1.0,  1.3,  0.9]
        elif Slag==0 and FlyAsh>0 and SiO2==0:
            return [1.0,  1.0,  1.4,  0.9]            
        elif Slag>0 and FlyAsh>0 and SiO2==0:
            return [1.2,  1.0,  3.5,   1.2]
        elif Slag==0 and FlyAsh==0 and SiO2>0:
            return [1.0,  1.0,  1.4,  0.9]
        else:
            return [1.0,  1.0,  1.0,  1.0]  



#Table5------------------------------------------------------------------------
def Table5_cal(cement,curing):
    if cement==1:
        if curing==1:
            return [1.0, 0.95]
        elif curing==3:
            return [4.0, 0.85]        
    elif cement==2:
        if curing==1:
            return [1.0, 0.95]
        elif curing==3:
            return [4.0, 0.85]
    elif cement==3:
        if curing==1:
            return [0.7, 0.98]
        elif curing==3:
            return [2.3, 0.92]

           
#Connect to SQL server
#server='DESKTOP-64NK71L\SQLEXPRESS'
server='DESKTOP-APU3DKF\SQLEXPRESS'
database='SCC_Strain'
#username='user'
#password='12345'
#cnxn=pyodbc.connect('Driver={ODBC Driver 13 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)
#cnxn=pyodbc.connect('Driver={ODBC Driver 13 for SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;')
cnxn=pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;')
cursor=cnxn.cursor()

Args=" Creep_type LIKE 'total' and Elastic_strain='yes'\
AND Coarse_agg>0 AND Fine_agg>0\
AND fc28>=17 and fc28<=110\
AND Tcur>=20 and Tcur<=30\
AND vs>11.3\
AND Cem_type=1"\
#AND Slag_percent=0 and FlyAsh_percent>0 and SiO2_percent=0"\
#AND Region NOT LIKE 'Taiwan'"\
#AND sigma_fct0>0.4"
#AND t0>28"

#AND Slag_percent=0 and FlyAsh_percent>0 and SiO2_percent=0"\
#AND Slag_percent=0 AND FlyAsh_percent=0 AND SiO2_percent=0"\
#AND Slag_percent>0 and FlyAsh_percent=0 and SiO2_percent=0"\
#
#AND Slag_percent>0 and FlyAsh_percent>0 and SiO2_percent=0"\
#AND Slag_percent=0 and FlyAsh_percent=0 and SiO2_percent>0"\


#Read data from SQL server
A="SELECT * FROM [SCC_Strain].[dbo].[SCC_Creep_Data] WHERE CreepFile IN(SELECT File_data_reference FROM [SCC_Strain].[dbo].[SCC_Creep_Info] WHERE "+Args+")"
B="Select * FROM [SCC_Strain].[dbo].[SCC_Creep_Info] WHERE " +Args
                         
#Read SQL query or database table into a DataFrame              
data=pd.read_sql(A,cnxn)
info=pd.read_sql(B,cnxn)
#Prepares and executes SQL
cursor.execute(A)
exdata=cursor.fetchall()
cursor.execute(B)
parameter=cursor.fetchall()

#range(start, stop[, step]) 
count=[None]*len(exdata)
for m in range(0,(len(exdata)-1)):
    if exdata[m+1][2]!=exdata[m][2]:
        count[m]=m
count[(len(exdata)-1)]=len(exdata)-1
count=list(filter(None,count))

count2=[None]*len(parameter)*2
for n in range(0,(len(parameter))):
    count2[2*n+1]=count[n]
for n in range(1,len(parameter)):
    count2[2*n]=count[n-1]+1
count2[0]=0

count3=[None]*len(parameter)
for n in range(0,len(parameter)):
    count3[n]=range(count2[2*n],count2[2*n+1]+1) 

time=[None]*len(parameter)
for n in range (0,(len(parameter))):
    time[n]=[None]*(count2[2*n+1]-count2[2*n]+1)
time=np.array(time)

for n in range(0,len(parameter)):
    for m in count3[n]:
        exdata[m][3]=float(exdata[m][3])
        time[n][count3[n].index(m)]=exdata[m][3]

value=[None]*len(parameter)
for n in range (0,(len(parameter))):
    value[n]=[None]*(count2[2*n+1]-count2[2*n]+1)
value=np.array(value)

for n in range(0,len(parameter)):
    for m in count3[n]:
        exdata[m][4]=float(exdata[m][4])
        value[n][count3[n].index(m)]=exdata[m][4]

#Set variable value
IDlist=[None]*len(parameter)
Jlist=[None]*len(parameter)
datalist=[None]*len(parameter)
tlist=[None]*len(parameter)
NOmixture=0
NOdatapoint=0
N=len(parameter)
ndlist=[0]*len(parameter)
nklist=[None]*len(parameter)
wijlist=[None]*len(parameter)
nwlist=[None]*len(parameter)
wijOij=[None]*len(parameter)
wijCijOij=[None]*len(parameter)
Ojlist=[None]*len(parameter)
wjlist=[None]*len(parameter)
wj2list=[None]*len(parameter)
fjlist=[0]*6
nlist=[0]*6
Fjlist=[0]*6
Milist=[0]*6
Mitemp=[0]*6
Oilist=[0]*6
Oitemp=[0]*6
Vilist=[0]*6
Vitemp=[0]*6
Ojlist_G=[0]*7
Ojtemp=[0]*7
RMSjlist=[0]*7
RMSjtemp=[0]*7
nGlist=[0]*7
FCEB=0
MCEB=0
VCEB=0
NCEB=0
Obar=0
RMSbar=0
wG=0
NG=0
q1list=[None]*len(parameter)
data=[None]*len(parameter)
for n in range(0,len(parameter)):
    data[n]=[None]*24


#Parameters of SQL file
for n in range(0,len(parameter)):
    ID=parameter[n][2]    
    IDlist[n]=ID
    exvalue=value[n]    

    wcm=parameter[n][10]
    wcm=float(wcm)
    acm=parameter[n][11]
    acm=float(acm)
# cemnetitious material content
    cm=parameter[n][15]
    cm=float(cm)

    SiO2=parameter[n][21]
    SiO2=float(SiO2)
    FlyAsh=parameter[n][22]
    FlyAsh=float(FlyAsh)
    Slag=parameter[n][23]
    Slag=float(Slag)
    NaturalPozzolan=parameter[n][24]
    NP=float(NaturalPozzolan)
    Filler=parameter[n][25]
    Filler=float(Filler)
     
# =============================================================================
#     SG=(c*Slag/100/(1-(FlyAsh+SiO2)/100))/(1-Slag/100/(1-(FlyAsh+SiO2)/100))
#     F=(c*FlyAsh/100/(1-(Slag+SiO2)/100))/(1-FlyAsh/100/(1-(Slag+SiO2)/100))
#     Si=(c*SiO2/100/(1-(Slag+FlyAsh)/100))/(1-SiO2/100/(1-(Slag+FlyAsh)/100))
#     admixture=Slag+FlyAsh+SiO2
#     c=c+SG+F+Si
# =============================================================================

    fc=parameter[n][39]
    fc=float(fc)

    D=parameter[n][47]*2
    D=float(D)

    tp=parameter[n][49]
    t0=parameter[n][50]
    Tcur=parameter[n][51]
    T=parameter[n][52]
    Heat=parameter[n][53]

    RH=parameter[n][54]
    #RH=RH.replace("%", "")
    RH=float(RH)

    curing=parameter[n][55]
    if curing==2:
        curing=3
    geo=parameter[n][56]
    cement=parameter[n][57]
    aggtype=parameter[n][58]

    regiontype=parameter[n][59]
    rtype=float(regiontype)
 
    sigma_fct0=parameter[n][60]       
    fineagg=parameter[n][70]
    coarseagg=parameter[n][71]
    fineagg=float(fineagg)    
    ca=float(coarseagg)    
    sa=float(fineagg/(fineagg+ca))

    Flyashtype=parameter[n][74]
    FAtype=float(Flyashtype)

    dt=parameter[n][75]
    dt=float(dt)
    
    w=wcm*cm

    ktaua=ktaua_cal(aggtype)
    kea=kea_cal(aggtype)
   
    Kh=Kh_cal(RH)
    Ks=Ks_cal(geo)
    
    BTh=BTh_cal(Tcur)
    BTs=BTs_cal(T)
    BTc=BTc_cal(T)

    t=time[n]
    t=np.array(t)+tp    
    t0wave=t0wave_cal(t0,BTh)
    tphat=tphat_cal(t0,tp,BTh,BTs)

    Table1=Table1_cal(cement)

    Table3=Table3_cal(cement)
    Table4S=Table4S_cal(Slag,FlyAsh,SiO2,NP,wcm,rtype)
    Table5=Table5_cal(cement,curing)
    Table4C=Table4C_cal(Slag,FlyAsh,SiO2,wcm,rtype)   
    


    taucem=Table1[0]
    ptaua=Table1[1]
    ptauw=Table1[2]
    ptauc=Table1[3]
    ecem=Table1[4]
    pea=Table1[5]
    pew=Table1[6]
    pec=Table1[7]
    #pesa=Table1[8] # Sandrate
    peca=Table1[8] # Coarse aggregate

    p1=Table3[0]
    p2=Table3[1]
    p3=Table3[2]
    p4=Table3[3]
    p5=Table3[4]
    p5H=Table3[5]
    p2w=Table3[6]
    p2c=Table3[7]
    #p2sa=Table3[8] # Sandrate    
    p2ca=Table3[8] # Coarse aggregate
    p3a=Table3[9]
    p3w=Table3[10]
    p4a=Table3[11]
    p4w=Table3[12]
    p5e=Table3[13]
    p5a=Table3[14]
    p5w=Table3[15]

    fshcem=Table4S[2]
    fshtau=Table4S[3]
    
    a=Table5[0]
    b=Table5[1]    
    
    fp2=Table4C[0]
    fp3=Table4C[1]   
    fp4=Table4C[2]
    fp5=Table4C[3]


#calculation starts here-------------------------------------------------------

    tau0=tau0_cal(acm,wcm,cm,taucem,ptaua,ptauw,ptauc)
    taush=taush_cal(D,Ks,ktaua,tau0)
    #e0=e0_cal(acm,wcm,cm,ecem,pea,pew,pec,sa,pesa) # Sandrate
    e0=e0_cal(acm,wcm,cm,ecem,pea,pew,pec,ca,peca) # Coarse aggregate

    fcmtp=fcmtp_cal(a,b,fc,tp)
    Ecmtp=Ecmtp_cal(fcmtp,fc)
    E28=E28_cal(fc)
    E607_m=E_cal(7*BTh+600*BTs)
    Et0_taush_m=E_cal(t0wave+taush*BTs)
    esh8=esh8_cal(e0,kea,E607_m,Et0_taush_m,fshcem)

#arraying
    
    t0_array=np.array([t0]*len(t))
    tp_array=np.array([tp]*len(t))
    tphat_array=np.array([tphat]*len(t))
    taush_array=np.array([taush]*len(t))
    t0wave_array=np.array([t0wave]*len(t))
    twave_array=(t-t0_array)*BTs
    that_array=(tphat_array+(t-tp_array)*BTc)

    S=np.array(S_cal(twave_array,taush,fshtau))
    esh=esh_cal(esh8,S,Kh,curing)*1000000

    q1=q1_cal(p1,Ecmtp)
    q1_array=np.array([q1])
    #q2=q2_cal(p2,wcm,p2w,cm,p2c,fp2,sa,p2sa) # Sandrate
    q2=q2_cal(p2,wcm,p2w,cm,p2c,fp2,ca,p2ca) # Coarse aggregate
    q3=q3_cal(q2,p3,acm,wcm,p3a,p3w,fp3)
    q4=q4_cal(p4,acm,wcm,p4a,p4w,fp4)
    r=r_cal(tphat)
    Z=Z_cal(tphat,tphat_array,that_array)
    Z=np.array(Z)
    Qf=Qf_cal(tphat)
    Q=Q_cal(Qf,r,Z)
    q5=q5_cal(p5,acm,wcm,p5a,p5w,Kh,esh8,p5e,fp5)
    C0=C0_cal(q2,q3,q4,Q,that_array,tphat_array)*1000000
    tp0hat_array=tp0hat_array_cal(tphat_array,t0wave_array)
    H=H_cal(that_array,t0wave_array,RH)
    Hc=Hc_cal(tp0hat_array,t0wave_array,RH)
    Cd=np.zeros(len(t))
    def Cd_cal(q5,p5H,H,Hc,that_array,tp0hat_array):
        for m in range(0,len(t)):
            if that_array[m]>=tp0hat_array[m]:
                Cd[m]=(q5*(math.exp(-1*p5H*H[m])-math.exp(-1*p5H*Hc[m]))**0.5)*1000000
            else:
                Cd[m]=0
    Cd_cal(q5,p5H,H,Hc,that_array,tp0hat_array)
    J=q1*1000000+C0+Cd
    q1list[n]=q1_array
    J=np.array(J,dtype=float)
    exvalue=np.array(exvalue,dtype=float)
    #if J[0]-exvalue[0]>=25:
    #    print(ID)
    z=np.polyfit(exvalue,J,1)
    p=np.poly1d(z)
    if z[0]<0.00001:
        #print 'Exceed lower bond'
        #print ID
        #print 'Predicted Value:'
        #print J
        #print z[0]
        Jlist[n]=np.array([0])
        datalist[n]=np.array([0])
    
   
    else:
        nj=float(len(t))
        wijOij[n]=[None]*len(t)
        if max(t-tp)<10:
            ndlist[n]=1
            nklist[n]=len(t)
            if nklist[n] != 0:
                wijlist[n][0]=nj/ndlist[n]/nklist[n]
            nwlist[n]=wijlist[n]
        elif 10<=max(t-tp)<100:
            ndlist[n]=2
            nklist[n]=[0,0]
            nwlist[n]=np.array([0,0])
            wijlist[n]=[0,0]
            wijOij[n]=[0]*len(t)
            wijCijOij[n]=[0]*len(t)
            for m in range(0,len(t)):
                if t[m]-tp<10:
                    nklist[n][0]=nklist[n][0]+1
                else:
                    nklist[n][1]=nklist[n][1]+1
            for o in range(0,2):
                if nklist[n][o] != 0:
                    wijlist[n][o]=nj/ndlist[n]/nklist[n][o]
            for m in range(0,2):
                nwlist[n]=nwlist[n]+wijlist[n][m]*nklist[n][m]
            for m in range(0,len(t)):
                if t[m]-tp<10:
                    wijOij[n][m]=wijlist[n][0]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][0]*(J[m]-exvalue[m])**2
                else:
                    wijOij[n][m]=wijlist[n][1]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][1]*(J[m]-exvalue[m])**2
            Ojlist[n]=sum(wijOij[n])/nwlist[n][0]
            wjlist[n]=(sum(wijCijOij[n])/(nj-1))**0.5/Ojlist[n]
            wj2list[n]=np.array(wjlist[n])**2
        elif 100<=max(t-tp)<1000:
            ndlist[n]=3
            nklist[n]=[0,0,0]
            nwlist[n]=np.array([0,0,0])
            wijlist[n]=[0,0,0]
            wijOij[n]=[0]*len(t)
            wijCijOij[n]=[0]*len(t)
            for m in range(0,len(t)):
                if t[m]-tp<10:
                    nklist[n][0]=nklist[n][0]+1
                elif 10<=t[m]-tp<100:
                    nklist[n][1]=nklist[n][1]+1
                else:
                    nklist[n][2]=nklist[n][2]+1
            for o in range(0,3):
                if nklist[n][o] != 0:
                    wijlist[n][o]=nj/ndlist[n]/nklist[n][o]
            for m in range(0,3):
                nwlist[n]=nwlist[n]+wijlist[n][m]*nklist[n][m]
            for m in range(0,len(t)):
                if t[m]-tp<10:
                    wijOij[n][m]=wijlist[n][0]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][0]*(J[m]-exvalue[m])**2
                elif 10<=t[m]-tp<100:
                    wijOij[n][m]=wijlist[n][1]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][1]*(J[m]-exvalue[m])**2
                else:
                    wijOij[n][m]=wijlist[n][2]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][2]*(J[m]-exvalue[m])**2
            Ojlist[n]=sum(wijOij[n])/nwlist[n][0]
            wjlist[n]=(sum(wijCijOij[n])/(nj-1))**0.5/Ojlist[n]
            wj2list[n]=np.array(wjlist[n])**2
        elif max(t-tp)>=1000:
            ndlist[n]=4
            nklist[n]=[0,0,0,0]
            nwlist[n]=np.array([0,0,0,0])
            wijlist[n]=[0,0,0,0]
            wijOij[n]=[0]*len(t)
            wijCijOij[n]=[0]*len(t)
            for m in range(0,len(t)):
                if t[m]-tp<10:
                    nklist[n][0]=nklist[n][0]+1
                elif 10<=t[m]-tp<100:
                    nklist[n][1]=nklist[n][1]+1
                elif 100<=t[m]-tp<1000:
                    nklist[n][2]=nklist[n][2]+1
                else:
                    nklist[n][3]=nklist[n][3]+1
            for o in range(0,4):
                if nklist[n][o] != 0:
                    wijlist[n][o]=nj/ndlist[n]/nklist[n][o]
            for m in range(0,3):
                nwlist[n]=nwlist[n]+wijlist[n][m]*nklist[n][m]
            for m in range(0,len(t)):
                if t[m]-tp<10:
                    wijOij[n][m]=wijlist[n][0]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][0]*(J[m]-exvalue[m])**2
                elif 10<=t[m]-tp<100:
                    wijOij[n][m]=wijlist[n][1]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][1]*(J[m]-exvalue[m])**2
                elif 100<=t[m]-tp<1000:
                    wijOij[n][m]=wijlist[n][2]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][2]*(J[m]-exvalue[m])**2
                else:
                    wijOij[n][m]=wijlist[n][3]*exvalue[m]
                    wijCijOij[n][m]=wijlist[n][3]*(J[m]-exvalue[m])**2
            Ojlist[n]=sum(wijOij[n])/nwlist[n][0]
            wjlist[n]=(sum(wijCijOij[n])/(nj-1))**0.5/Ojlist[n]
            wj2list[n]=np.array(wjlist[n])**2
        for m in range(0,len(t)):
            if t[m]-tp<=10:
                if exvalue[m]!=0:
                    fjlist[0]=fjlist[0]+(((J[m]-exvalue[m])/exvalue[m])*100)**2
                    Oitemp[0]=Oitemp[0]+exvalue[m]
                    Vitemp[0]=Vitemp[0]+(J[m]-exvalue[m])**2
                    nlist[0]=nlist[0]+1
                    Mitemp[0]=Mitemp[0]+J[m]/exvalue[m]
            elif 10<t[m]-tp<=100:
                fjlist[1]=fjlist[1]+(((J[m]-exvalue[m])/exvalue[m])*100)**2
                Mitemp[1]=Mitemp[1]+J[m]/exvalue[m]
                Oitemp[1]=Oitemp[1]+exvalue[m]
                Vitemp[1]=Vitemp[1]+(J[m]-exvalue[m])**2
                nlist[1]=nlist[1]+1
            elif 100<t[m]-tp<=365:
                fjlist[2]=fjlist[2]+(((J[m]-exvalue[m])/exvalue[m])*100)**2
                Mitemp[2]=Mitemp[2]+J[m]/exvalue[m]
                Oitemp[2]=Oitemp[2]+exvalue[m]
                Vitemp[2]=Vitemp[2]+(J[m]-exvalue[m])**2
                nlist[2]=nlist[2]+1
            elif 365<t[m]-tp<=730:
                fjlist[3]=fjlist[3]+(((J[m]-exvalue[m])/exvalue[m])*100)**2
                Mitemp[3]=Mitemp[3]+J[m]/exvalue[m]
                Oitemp[3]=Oitemp[3]+exvalue[m]
                Vitemp[3]=Vitemp[3]+(J[m]-exvalue[m])**2
                nlist[3]=nlist[3]+1
            elif 730<t[m]-tp<=1095:
                fjlist[4]=fjlist[4]+(((J[m]-exvalue[m])/exvalue[m])*100)**2
                Mitemp[4]=Mitemp[4]+J[m]/exvalue[m]
                Oitemp[4]=Oitemp[4]+exvalue[m]
                Vitemp[4]=Vitemp[4]+(J[m]-exvalue[m])**2
                nlist[4]=nlist[4]+1
            elif t[m]-tp>1095:
                fjlist[5]=fjlist[5]+(((J[m]-exvalue[m])/exvalue[m])*100)**2
                Mitemp[5]=Mitemp[5]+J[m]/exvalue[m]
                Oitemp[5]=Oitemp[5]+exvalue[m]
                Vitemp[5]=Vitemp[5]+(J[m]-exvalue[m])**2
                nlist[5]=nlist[5]+1
        for m in range(0,len(t)):
            if 3<=t[m]-tp<=9.9:
                if exvalue[m]!=0:
                    Ojtemp[0]=Ojtemp[0]+exvalue[m]
                    RMSjtemp[0]=RMSjtemp[0]+(J[m]-exvalue[m])**2
                    nGlist[0]=nGlist[0]+1
            elif 9.9<t[m]-tp<=31.5:
                Ojtemp[1]=Ojtemp[1]+exvalue[m]
                RMSjtemp[1]=RMSjtemp[1]+(J[m]-exvalue[m])**2
                nGlist[1]=nGlist[1]+1
            elif 31.5<t[m]-tp<=99:
                Ojtemp[2]=Ojtemp[2]+exvalue[m]
                RMSjtemp[2]=RMSjtemp[2]+(J[m]-exvalue[m])**2
                nGlist[2]=nGlist[2]+1
            elif 99<t[m]-tp<=315:
                Ojtemp[3]=Ojtemp[3]+exvalue[m]
                RMSjtemp[3]=RMSjtemp[3]+(J[m]-exvalue[m])**2
                nGlist[3]=nGlist[3]+1
            elif 315<t[m]-tp<=999:
                Ojtemp[4]=Ojtemp[4]+exvalue[m]
                RMSjtemp[4]=RMSjtemp[4]+(J[m]-exvalue[m])**2
                nGlist[4]=nGlist[4]+1
            elif 999<t[m]-tp<=3159:
                Ojtemp[5]=Ojtemp[5]+exvalue[m]
                RMSjtemp[5]=RMSjtemp[5]+(J[m]-exvalue[m])**2
                nGlist[5]=nGlist[5]+1
            elif t[m]-tp>=3159:
                Ojtemp[6]=Ojtemp[6]+exvalue[m]
                RMSjtemp[6]=RMSjtemp[6]+(J[m]-exvalue[m])**2
                nGlist[6]=nGlist[6]+1

#plot to analysis
        '''if fc<=55:
            lg1=plot.scatter(exvalue,J,c='b',s=10)
        if fc>55:
            lg2=plot.scatter(exvalue,J,c='r',s=10)'''
        
        '''if rtype==1:
            lg1=plot.scatter(exvalue,J,c='r',s=10)
        if rtype>1:
            lg2=plot.scatter(exvalue,J,c='b',s=10)'''
            
        '''if FAtype<=1:
            lg1=plot.scatter(exvalue,J,c='b',s=10)
        if FAtype>1:
            lg2=plot.scatter(exvalue,J,c='r',s=10)'''

        '''if dt<=90:
            lg1=plot.scatter(exvalue,J,c='b',s=10)
        if dt>90:
            lg2=plot.scatter(exvalue,J,c='r',s=10)'''
        
        if t0<=7:
            lg1=plot.scatter(exvalue,J,c='b',s=10)
        if t0>7:
            lg2=plot.scatter(exvalue,J,c='r',s=10)            

    #plot.scatter(exvalue,J,c='b',s=10)    

# =============================================================================
#     if exvalue[-1]<50:
#         print(ID)
#     #print(J[0])
#     '''for i in range(len(t)):
#         if (abs(exvalue[i])-abs(J[i]))>=70:
#                 print(ID)'''
# =============================================================================
    NOmixture=NOmixture+1
    NOdatapoint=NOdatapoint+len(J)
    plot.xlim(0,300)
    plot.ylim(0,300)
    plot.axes().set_aspect('equal', adjustable='box')
    Jlist[n]=J
    datalist[n]=exvalue
    tlist[n]=t-tp



wbp=(sum(wj2list)/N)**0.5

for n in range(0,6):
    if nlist[n]==0:
        Fjlist[n]=0
    elif nlist[n]==1:
        Fjlist[n]=fjlist[n]/(nlist[n])
    else:
        Fjlist[n]=fjlist[n]/(nlist[n]-1)

for n in range(0,6):
    if nlist[n]==0:
        Milist[n]=0
        Oilist[n]=0
    else:
        Milist[n]=Mitemp[n]/nlist[n]
        Oilist[n]=Oitemp[n]/nlist[n]

for n in range(0,6):
    if nlist[n]==0:
        Vilist[n]=0
    elif nlist[n]==1:
        Vilist[n]=(((Vitemp[n]/nlist[n])**0.5)/Oilist[n])**2
    else:
        Vilist[n]=(((Vitemp[n]/(nlist[n]-1))**0.5)/Oilist[n])**2

for n in range(0,6):
    if nlist[n]!=0:
        NCEB=NCEB+1
        
for n in range(0,7):
    if nGlist[n]==0:
        Ojlist_G[n]=0
    else:
        Ojlist_G[n]=Ojtemp[n]/nGlist[n]

for n in range(0,7):
    if nGlist[n]==0:
        RMSjlist[n]=0
    elif nGlist[n]==1:
        RMSjlist[n]=(RMSjtemp[n]/nGlist[n])**0.5
    else:
        RMSjlist[n]=(RMSjtemp[n]/(nGlist[n]-1))**0.5
        
for n in range(0,7):
    if nGlist[n]!=0:
        NG=NG+1

Obar=sum(Ojlist_G)/NG
RMSbar=sum(RMSjlist)/NG
wG=RMSbar/Obar
FCEB=(sum(Fjlist)/NCEB)**0.5
MCEB=sum(Milist)/NCEB
VCEB=(sum(Vilist)/NCEB)**0.5
    
Jlist2=[]
for n in range(0,len(parameter)):
    if sum(Jlist[n])>0:
        Jlist2.extend(Jlist[n])

datalist2=[]
for n in range(0,len(parameter)):
    if sum(datalist[n])>0:
        datalist2.extend(datalist[n]) 
        
tlist2=[]
for n in range(0,len(parameter)):
    if sum(datalist[n])>0:
        tlist2.extend(tlist[n]) 
q1list2=[]
for n in range(0,len(parameter)):
    q1list2.extend(q1list[n]) 

q1array=np.array(q1list2)*1e6

average=sum(datalist2)/len(datalist2)
SST=sum((np.array(datalist2)-average)**2)
SSR=sum((np.array(datalist2)-np.array(Jlist2))**2)
Rsquare=1-SSR/SST


if Rsquare<0:
    Rsquare='div'
else:
    Rsquare=round(Rsquare,2)


def upper(x):
    return 1/0.6*x
def lower(x):
    return 0.6*x
x=np.arange(0,1000,1)

from sklearn.metrics import mean_squared_error
MSE_pred=round(mean_squared_error(datalist2, Jlist2),2)
NRMSE=MSE_pred**0.5/(np.quantile(datalist2,0.75)-np.quantile(datalist2,0.25))*100
NRMSE=round(NRMSE,2)
label_upper="Test"
plot.plot(x,x,color='k')
plot.plot(x,upper(x),"r--")
plot.plot(x,lower(x),"r--")
#plot.text(125, 175, r'+40%', fontsize=12,color='r')
#plot.text(170, 100, r'-40%', fontsize=12,color='r')
#plot.text(15, 180, 'NO. of mixtures: %s\n\nNO. of Data Points: %s\n\nR Square: %s\n\nMSE: %s '%(len(parameter), len(exdata),Rsquare,MSE_pred),style='italic',bbox={'facecolor':'white', 'alpha':0.5,'pad':8})
plot.text(15, 180, 'NO. of mixtures: %s\n\nNO. of Data Points: %s\n\n$R^2$: %s\n\nNRMSE: %s '%(len(parameter), len(exdata),Rsquare,NRMSE),style='italic',bbox={'facecolor':'white', 'alpha':0.5,'pad':8})
plot.title('Model B4TW-SCC Total Creep',fontsize=15)
plot.xlabel(r'Test Data ($10^{-6}$/MPa)',fontsize=12)
plot.ylabel('Predicted Value ($10^{-6}$/MPa)',fontsize=12)

#legend2=plot.legend([lg1,lg2],['Taiwan','Others'],loc='lower right',scatterpoints=1,edgecolor='k',fontsize=10)
#legend2=plot.legend([lg1,lg2],['$f_c \leq$ 55MPa','$f_c >$ 55MPa'],loc='lower right',scatterpoints=1,edgecolor='k',fontsize=10)
#legend2=plot.legend([lg1,lg2],['Type F','Type C'],loc='lower right',scatterpoints=1,edgecolor='k',fontsize=10)
legend2=plot.legend([lg1,lg2],['$t_0 \leq$ 7 days','$t_0 >$ 7 days'],loc='lower right',scatterpoints=1,edgecolor='k',fontsize=10)

# =============================================================================
# plot.show()
# from sklearn.metrics import mean_squared_error
# MSE_pred=round(mean_squared_error(datalist2, Jlist2),2)
# print(MSE_pred)
# NRMSE=MSE_pred**0.5/(np.quantile(datalist2,0.75)-np.quantile(datalist2,0.25))*100
# NRMSE=round(NRMSE,2)
# fig = plot.figure(figsize=(4,2))
# ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
# #fig,ax=plot.subplots(111, frameon=False, xticks=[], yticks=[])
# data_table=([['value'],[NOmixture],[int(NOdatapoint)],[Rsquare],np.round([wbp*100],2),
#              np.round([FCEB],2),np.round([MCEB],2),np.round([VCEB*100],2),np.round([wG*100 ],2),[MSE_pred],[NRMSE]])
# 
# rows=('Model B4TW-SCC2022','Number of Data Sets','Number of Data points','Rsquare','w_BP(%)','F_CEB(%)','M_CEB','V_CEB(%)','w_G(%)','MSE','NRMSE(%)')
# the_table = plot.table(cellText=(data_table),rowLabels=rows, loc='center',cellLoc='center')
# the_table.auto_set_font_size(False)
# the_table.set_fontsize(15)
# the_table.scale(0.5, 2)
# =============================================================================


