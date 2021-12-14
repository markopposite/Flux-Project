
# coding: utf-8

# In[2]:


# import os
# os.getcwd()


# In[2]:


import sys
sys.path.append('/home/jovyan/work/shared/')


# In[3]:


import nanograv_data as api
from pypulse import Archive
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import os


# In[11]:


### Name of pulsar 
# name = '0613-0200'


# In[4]:


def database(pulsar_name):
    """
    Input a pulsar name '1234-5678'
    Builds a dictionary of pulsar data:
    'Name','Mean Flux','Subint Fluxes','Freq','MJD','Subint','Chan','Bin','Polarity','Period','Year'
    """
    
    name = pulsar_name
    
    pith = ' ~ /nanograv/data/'+name+'/guppi/*/processed/*guppi_*_J'+name+'_*.11y.x.calib'
    poth = '/home/jovyan/work/shared/Flux-Project/data/'+name+'.txt'
    os.system('ls'+pith+' > '+poth)
    stu = 'ls '+pith+' > '+poth
    print stu


    phil = name+'.txt'
    phil_collins =[]

    with open('data/'+phil,'rb') as phil:
        contents=phil.readlines()
        for item in contents:
            if '/nanograv' not in item:
                continue
            y = item.split('\n')
            phil_collins.append(y[0])

    print len(phil_collins)        
    # print phil_collins


    pulsar_dict={}

    if os.path.exists('data/'+name):
        GG = pickle.load(file('data/'+name))
        print len(GG)
    else:
        GG=pulsar_dict
    for filename in phil_collins:
        if filename in GG:
            pass
        elif filename not in GG:
            try:
                ar=Archive(filename)
                print filename

                l = ar.getCenterFrequency()
                i = ar.getNsubint()
                h = ar.getNchan()
                g = ar.getNbin()
                f = ar.getNpol()
                e = ar.getPeriod()

                ar.pscrunch()
                ar.fscrunch()
                ar.bscrunch()
    #             ar.tscrunch()
    #             m = ar.data
    #             n = m[0][0][0][0] #average flux from full tscrunch       

                ar.tscrunch(nsubint=8)
                m = ar.data 
                n = np.mean(m) # Average flux from subints            

                j = ar.getMJD(full=True)

                x = filename
                y = x.split('/')
                o = y[7] # filename
                r = y[5] #year

                new_key = o
                new_value=[]
                new_value.extend([n,m,l,j,i,h,g,f,e,r])
                pulsar_dict[new_key]=new_value

            except:
                pass

        if len(pulsar_dict) % 10 == 0:
            GG.update(pulsar_dict)
            with open('data/'+name,'wb') as wfp:
                pickle.dump(GG, wfp)

    GG.update(pulsar_dict)
    with open('data/'+name,'wb') as wfp:
        pickle.dump(GG, wfp)
    # with open(nahme, 'rb') as rr:
    #     HH = pickle.load(file(nahme))
    # print len(HH)
    print len(GG)
    print len(pulsar_dict)


    csv_filename = name+'.csv'
    with open('data/'+csv_filename,'w') as x:
        writer = csv.writer(x)
        writer.writerow(["Name","Mean Flux","Subint Fluxes","Freq","MJD","Subint","Chan","Bin","Polarity","Period","Year"])
        for key, value in GG.items():
            z = GG[key]
            writer.writerow([key,z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7],z[8],z[9]])


# In[ ]:





# In[ ]:





# In[1]:


# print name
# print len(pulsar_dict)
# # print pulsar_dict


# In[6]:


# data = api.get_processed_files(limit=200,pulsar="J1713+0747",mjd=56380) # ,profile_format='PSRFITS')
# filename = data['processed_file_location'][0]
# print(data)

