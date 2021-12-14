import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as io
from astropy.table import Table as table
    
"""Invariant parameters for NE2001, all frequencies: gl,gb,dm,dist_kpc,log_sm,trans_freq,em

    820 MHz parameters: ang_br_820,puls_br_820,scint_bw_820,scint_time_820

    1500 parameters: ang_br_1500,puls_br_1500,scint_bw_1500,scint_time_1500

    820 MHz stats from flux code: mean_820,std_820,mn_820,mx_820,length_820,
                                    med_820,mod_820,a_820,b_820,dof_820

    1500 MHz stats from flux code: mean_1500,std_1500,mn_1500,mx_1500,
                                    length_1500,med_1500,mod_1500,a_1500,b_1500,dof_1500
"""

        
def log_fit(x, y):
    """Fits a line to the data for a log-log plot, returns slope, intercept
        Set plot=True to show"""
    log_x = np.log10(x)
    log_y = np.log10(y)
    linear = np.polyfit(log_x,log_y,1)
    lin = np.polyfit(x,y,1)
    a1,b1 = lin[0],lin[1]
    line1 = a1*x + b1
    a,b = linear[0],linear[1]
    line = a*log_x+b
    h = []
    h.append(a),h.append(b),h.append(log_x),h.append(log_y),h.append(line)
    h.append(a1),h.append(b1),h.append(line1)
    
    return h


def plot(x,y,xlab,ylab,clr,titl,log=False):
    fit = log_fit(x,y)
    ft = 5
    if log:
        print (fit[0]) #,fit[1])
        x,y = fit[2],fit[3]
        plt.xlabel('log(' + xlab + ')',fontsize=ft)
        plt.ylabel('log(' + ylab + ')',fontsize=ft)
        plt.scatter(x,y,marker='+',linewidth=.8,color=clr)
        plt.plot(fit[2],fit[4],linewidth=.8)
    else:
        #print (fit[5],fit[6])
        plt.xlabel(xlab,fontsize=ft)
        #plt.ylabel('log(' + ylab + ')',fontsize=ft)
        plt.ylabel(ylab,fontsize=ft)
        plt.scatter(x,y,marker='+',linewidth=.8,color=clr)
        #plt.plot(x,fit[7],linewidth=.8)
    plt.title(titl,fontsize=6)
    plt.grid(linestyle='dotted')

    
def correlator(x, y, hi_res=True):
    """ Enter two equal sized arrays for x and y
        Values must be strings matching column headings
        If log=False, the plot is x and logy, if True, plot is logx and logy
        Depends on the plot() and log_fit() functions
    """
    
    ne15 = io.read('NE2001_1500.csv',data_start=2)
    ne8 = io.read('NE2001_820.csv',data_start=2)
    dist15 = io.read('Dist_1500.csv',data_start=2)
    dist8 = io.read('Dist_820.csv',data_start=2)
    atnf = io.read('atnf.csv', data_start=2)
    
    xname = x
    yname = y
    
    if y in dist8.colnames:
        y1 = dist8[y]
        y2 = dist15[y]
    #elif x in atnf.colnames:
    #    y1 = atnf[y]
    #    y2 = atnf[y]
    else:
        y1 = ne8[y]
        y2 = ne15[y]

    if x in dist8.colnames:
        x1 = dist8[x]
        x2 = dist15[x]
   # elif x in atnf.colnames:
   #     x1 = atnf[x]
   #     x2 = atnf[x]
    else: 
        x1 = ne8[x]
        x2 = ne15[x]

    plt.subplot(221)
    plot(x1,y1,xname,yname,log=False,clr='red',titl='820 MHz linear-log')

    plt.subplot(222)
    plot(x1,y1,xname,yname,log=True,clr='k',titl='820 MHz log-log')

    plt.subplot(223)
    plot(x2,y2,xname,yname,log=False,clr='red',titl='1500 MHz linear-log')
    
    plt.subplot(224)
    plot(x2,y2,xname,yname,log=True,clr='k',titl='1500 MHz log-log')
    
    plt.subplots_adjust(right=.8,hspace=.5,wspace=.3)
    plt.show()
    if hi_res:
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300                
           
        
def linear_fit(x,y,plot=True,labely=None,labelx=None,hi_res=False):
    """Fits a line to the data for a log-log plot, returns slope, intercept
        Set plot=True to show"""
    log_x = np.log10(x)
    log_y = np.log10(y)
    linear = np.polyfit(log_x,log_y,1)
    a,b = linear[0],linear[1]
    line = a*log_x+b
    h = []
    h.append(a),h.append(b)
    
    if plot:
        plt.scatter(x,y,marker='+',linewidth=.8,color='red')
        plt.ylabel(labely)
        plt.xlabel(labelx)
        plt.grid(linestyle='dotted')
        plt.show()
        
        plt.scatter(log_x,log_y,marker='+',linewidth=.8,color='k')
        plt.plot(log_x,line,linewidth=.8)
        plt.grid(linestyle='dotted')
        plt.ylabel(labely)
        plt.xlabel('log(' + labelx + ')')
#         plt.legend(prop={'size': 6})
        # plt.title(pulsar_name)
        plt.show()
        if hi_res:
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['figure.dpi'] = 300        
    return h


def data(x):
    
    params=[gl,gb,dm,dist_kpc,log_sm,trans_freq,em,ang_br_820,puls_br_820,scint_bw_820,scint_time_820
             ,ang_br_1500,puls_br_1500,scint_bw_1500,scint_time_1500, mean_820,std_820,mn_820,mx_820
             ,length_820,med_820,mod_820,a_820,b_820,dof_820, mean_1500,std_1500,mn_1500,mx_1500
             ,length_1500,med_1500,mod_1500,a_1500,b_1500,dof_1500]
    
    for item in params:
        if x in item:
            return item   

### Invariant parameters for NE2001, all frequencies
gl,gb,dm,dist_kpc,log_sm,trans_freq,em = np.loadtxt('NE2001_820.csv'
                    ,skiprows=2,delimiter=',',usecols=(1,2,3,4,5,10,11),unpack=True)


### 820 MHz parameters
ang_br_820,puls_br_820,scint_bw_820,scint_time_820 = np.loadtxt('NE2001_820.csv'
                    ,skiprows=2,delimiter=',',usecols=(6,7,8,9),unpack=True)


### 1500 parameters
ang_br_1500,puls_br_1500,scint_bw_1500,scint_time_1500 = np.loadtxt('NE2001_1500.csv',skiprows=2,delimiter=','
                    , usecols=(6,7,8,9),unpack=True)


### 820 MHz stats from flux code
mean_820,std_820,mn_820,mx_820,length_820,med_820,mod_820,a_820,b_820,dof_820 = np.loadtxt('Dist_820.csv'
                    ,skiprows=2,delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10),unpack=True)


### 1500 MHz stats from flux code
mean_1500,std_1500,mn_1500,mx_1500,length_1500,med_1500,mod_1500,a_1500,b_1500,dof_1500 = np.loadtxt('Dist_1500.csv',skiprows=2,delimiter=','
                    , usecols=(1,2,3,4,5,6,7,8,9,10),unpack=True)