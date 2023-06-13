#Question: unsure how to handle making sure events don't repeat and differentiating files. 
#Solution: Each upload will use a different i3 file. 

"""
IceCube Zooniverse Subject Maker & Uploader -- Compilation of functions until I can turn it into a package. 
- Is missing compression function, but Handbrake has a saved preset called Beta2_Compression
- Will need to manually queue in videos to have Handbrake compress them
"""
#Imports:
import subprocess
import numpy as np
import sys
import os
import argparse
import csv
from tables import *
import pandas as pd
from matplotlib import pyplot as plt
import h5py

#Zooniverse Imports
from panoptes_client import Panoptes, Project, SubjectSet, Subject
import magic
import glob

#IceCube imports:
#import nuflux -- some icecube packages don't work unless in cobalt. 
#import simweights
from icecube.icetray import I3Units
import icecube.MuonGun
from icecube import dataio, dataclasses, icetray, MuonGun
from I3Tray import *
from icecube.hdfwriter import I3HDFWriter
from mc_labeler import MCLabeler #make sure that mc_labeler script is in directory. 

#Just in case not already hard-coded in somewhere, here is the GCD. 

#note to self: need GCD in order to work. make sure this is updated to whatever location the gcd is. 
gcd = '/home/icecube/Desktop/eliz_zooniverse/icecubezooniverseproj_ver3/launch/Phase1/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

"""
Filter 1:
- Use in ice split only. 
- Add mc_labeler tray.
- HDFWriter to make hdf table of info.
This is definetly not the most efficient since it needs to run the mc labeler on all frames, but it is what it is for now ¯\_(ツ)_/¯
"""
def filter1(infile, outdir):
    drive, ipath =os.path.splitdrive(infile)
    path, ifn = os.path.split(ipath)
    infile_name = infile.split('/')[-1]
    tray = I3Tray()
    tray.Add('I3Reader', FilenameList=[infile])
    tray.Add(MCLabeler)
    tray.Add('I3Writer', 'EventWriter',
    FileName= outdir+'/mc_labeled_'+infile_name,
        Streams=[icetray.I3Frame.TrayInfo,
        icetray.I3Frame.Geometry,
        icetray.I3Frame.Calibration,
        icetray.I3Frame.DetectorStatus,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Physics,
        icetray.I3Frame.Stream('S')],
        DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddSegment(I3HDFWriter, Output = f'{outdir}mc_labeled_{infile_name}.hd5', Keys = ['I3EventHeader','I3MCWeightDict',\
    'ml_suite_classification','NuGPrimary','PoleMuonLinefit', 'PoleMuonLinefitParams', 'PoleMuonLlhFitMuE', 'PoleMuonLlhFitFitParams',\
    'PoleMuonLlhFit','PolyplopiaInfo','PolyplopiaPrimary','I3MCTree','I3MCTree_preMuonProp','classification'], SubEventStreams=['InIceSplit'])
    tray.AddModule('TrashCan','can')

    tray.Execute()
    tray.Finish()
"""
Filter 2:
- Pick out random events that have a uniform energy distribution.
- Cutoff at PeV. 
- Select for only truths 1, 2, 3, 4, 6, 7, 8 (see enums.py)
- Takes a random seed (will want to program this to be optional later on.)
- Is standard to use bin num = 25 to follow most IceCube energy histograms
"""
def uniformenergy_events(hdf,bin_num, size,random_seed,subrun = False):
    #open hdf of desired i3 file
    hdf = f'{hdf}'
    hdf_file = h5py.File(hdf, "r+")
    print(f"Number of Bins: {bin_num}\nNumber of Events per Bin: {size}\nTotal Number of Events: {size*bin_num}") #Get some quick info

    #turn hdf into pandas dataframe.  

    event_id = hdf_file['I3EventHeader']['Event'][:] #Event ID
    run_id = hdf_file['I3EventHeader']['Run'][:] #Run ID

    #classifier predictions
    pred_skim_val = hdf_file['ml_suite_classification']['prediction_0000'][:] #Skimming
    pred_cascade_val = hdf_file['ml_suite_classification']['prediction_0001'][:] #Cascade
    pred_tgtrack_val = hdf_file['ml_suite_classification']['prediction_0002'][:] #Through-Going Track
    pred_starttrack_val = hdf_file['ml_suite_classification']['prediction_0003'][:] #Starting Track
    pred_stoptrack_val = hdf_file['ml_suite_classification']['prediction_0004'][:] #Stopping Track

    truth_label = hdf_file['classification']['value'][:] #mc_labeler value

    energy_val = hdf_file['NuGPrimary']['energy'][:]
    zenith_val = hdf_file['NuGPrimary']['zenith'][:]
    
    ow = hdf_file['I3MCWeightDict']['OneWeight'][:]

    #Dataframe time
    df = pd.DataFrame(dict(run = run_id, event = event_id, truth_classification = truth_label, pred_skim = pred_skim_val, pred_cascade = pred_cascade_val,\
        pred_tgtrack = pred_tgtrack_val, pred_starttrack = pred_starttrack_val, pred_stoptrack = pred_stoptrack_val,energy = energy_val, zenith = zenith_val,\
        oneweight = ow))
    
    hdf_file.close() #close hdf file now that dataframe is made. 
    
    #get truth id label (i.e. as words) into dataframe
    
    #Dictionary of event truth labels according to mc_labeler and enums, not DNNClassifier
    label_dict = {0:'unclassified',
                        1:'throughgoing_track',
                        2:'starting_track',
                        3:'stopping_track',
                        4:'skimming_track',
                        5:'contained_track',
                        6:'contained_em_hadr_cascade',
                        7:'contained_hadron_cascade',
                        8:'uncontained_cascade',
                        9:'glashow_starting_track',
                        10:'glashow_electron',
                        11:'glashow_tau_double_bang',
                        12:'glashow_tau_lollipop',
                        13:'glashow_hadronic',
                        14:'throughgoing_tau',
                        15:'skimming_tau',
                        16:'double_bang',
                        17:'lollipop',
                        18:'inverted_lollipop',
                        19:'throughgoing_bundle',
                        20:'throughgoing_bundle',
                        21:'tau_to_mu'}
    
    #turn numbers into words!
    word_truth_labels = []
    for x in df['truth_classification']:
        word_truth_labels.append(label_dict[int(x)])
    label_str = f'#truth_classification_label'
    df[label_str] = word_truth_labels

    #add in other key-value pairs into dataframe

    #returns value of max ML score. 
    df['max_score_val'] = df[['pred_skim','pred_cascade','pred_tgtrack','pred_starttrack','pred_stoptrack']].max(axis='columns')

    #Getting largest ML prediction score. 

    #returns index of max ML score. 
    df['idx_max_score'] = df[['pred_skim','pred_cascade','pred_tgtrack','pred_starttrack','pred_stoptrack']].idxmax(axis='columns')
    

    #Pick events "manually", cut at PeV-ish.
    
    truth_vals_list = [1, 2, 3, 4, 6, 7, 8]

    df_masked = df[df['truth_classification'].isin(truth_vals_list)]

    df_filtered = df_masked.loc[(np.log10(df_masked['energy'][:]) <= 6)]
    print("Initial Number of Events:",len(df))
    print("Number of Events after filtering:",len(df_filtered))
    hist_df, bin_edges_df = np.histogram(np.log10(df_filtered['energy'][:]), density=False,bins=bin_num)
    #Use np.digitize to return index of the bin each energy value belongs to. 
    energy_bin_index = np.digitize(np.log10(df_filtered['energy']),bin_edges_df)
    #take any value less than fist bin is put in bin 1, and any value greater than 25th bin now in bin 25. 
    energy_bin_index = np.clip(energy_bin_index,1,bin_num) 
    
    #get bins and labels. 
    bins = bin_edges_df
    labels = np.arange(1,bin_num+1)

    df_filtered['binned_log10E'] = pd.cut(x = np.log10(df_filtered['energy']), bins = bins, labels = labels, include_lowest = True) #might throw an error?
    #Loop through each bin number and mask out any events not in that bin number.

    seed=random_seed
    rng = np.random.default_rng(seed)

    random_event_indices = np.array([]) #empty array for event indices

    for i in np.arange(1,bin_num+1):
        e_bin = df_filtered.loc[df_filtered['binned_log10E']== i]
        random_energy_index = rng.choice(e_bin.index,replace=False, size=size) #pick specified number of events from each energy bin. 
        random_event_indices = np.append(random_event_indices,random_energy_index)
    
    
    random_event_energies = df_filtered.loc[df_filtered.index.intersection(random_event_indices)]

    height_df_idx_max_score = random_event_energies['idx_max_score'].value_counts(sort=False) #for making histograms of most confident ML prediction.
    
    if subrun == True:
        print("Please input subrun number:")
        sub_id = input()
        csv_name = f'random_events_uniform_energy_distrib_{run_id[0]}_{sub_id}.csv'
    else:
        csv_name = f'random_events_uniform_energy_distrib_{run_id[0]}.csv'
    
    return random_event_energies.to_csv(csv_name)
    #return event_characterization_plots('random_events_uniform_energy_distrib.csv')
"""
Optional:
- Make characterization plots of random events
"""
def event_characterization_plots(event_csv):
    df = pd.read_csv(event_csv)
    
    if {'orig_idx','Unnamed: 0'}.issubset(df.columns):
        df = df.drop(['Unnamed: 0'], axis=1)
    else:
        df.rename({'Unnamed: 0': 'orig_idx'}, axis=1, inplace=True)#fix weird column name
        df.set_index('orig_idx')#reset index
    
    #making hist of max predictions
    skim_ml = len(df.loc[df['idx_max_score']=='pred_skim'])
    cascade_ml = len(df.loc[df['idx_max_score']=='pred_cascade'])
    tg_track_ml = len(df.loc[df['idx_max_score']=='pred_tgtrack'])
    start_track_ml = len(df.loc[df['idx_max_score']=='pred_starttrack'])
    stop_track_ml = len(df.loc[df['idx_max_score']=='pred_stoptrack'])

    counts= [skim_ml, cascade_ml, tg_track_ml, start_track_ml, stop_track_ml]
    names = ['pred_skim','pred_cascade','pred_tgtrack','pred_starttrack','pred_stoptrack']
    
    #Making Subplots
    fig, axss = plt.subplots(1,4,figsize=(15,5),facecolor='white', sharey=True)
    
    
    axs = np.ravel(axss)
    #labels = [r'log$_{10}$(E [1/GeV])', 'Truth Label', etc...]
    lw=3
    for idx, ax in enumerate(axs):
        cmap = plt.cm.get_cmap('plasma')
        color = cmap(idx * (1/len(axs))) #change color=color
        if idx == 0:
            ax.hist(np.log10(df['energy'][:]), bins = 25, color = cmap(idx * (1/len(axs))),density=False,histtype='step', lw=lw)
            ax.set_ylabel('Event Count')
            ax.set_xlabel(r'log$_{10}$(E [GeV])')
        if idx == 1:
            ax.hist(df['truth_classification'][:],histtype='step',density=False,lw=lw, color = cmap(idx * (1/len(axs))))
            ax.set_xlabel('Truth Label')
        if idx == 2:
            ax.bar(x=[0, 1, 2,3,4],height=counts,color = cmap(idx * (1/len(axs))))
            ax.set_xlabel('Classifier Label')
        if idx == 3:
            ax.hist(df['max_score_val'][:],bins=5,histtype='step',density=False, color=color, lw=lw)
            ax.set_xlabel('Classifier Score')
        ax.grid(True)
    """
    Note: currently this needs to be fixed, but including here just in case. This is not what I used for MS Thesis
    #Energy Histogram
    axs[0].hist(np.log10(df['energy'][:]),bins=25,histtype='step',density=False)
    #plt.semilogy()
    axs[0].set_ylabel('Event Count')
    axs[0].set_xlabel(r'log$_{10}$(E [GeV])')
    #axs[0].set_title("Log10(Energy) Histogram")

    #Truth Labels
    axs[1].hist(df['truth_classification'][:],histtype='stepfilled',density=False, align='right')
    #axs[1].set_ylabel('Event Count')
    #axs[1].set_xlabel('Truth Label')
    axs[1].set_xticks([0, 1, 2,3,4,5,6,7,8,9])
    mc_labeler_labels = ['unclassified','throughgoing_track','starting_track','stopping_track',\
                         'skimming_track','contained_track','contained_em_hadr_cascade',\
                         'contained_hadron_cascade','uncontained_cascade']
    axs[1].set_xticklabels(mc_labeler_labels,rotation=90)
    #axs[1].set_title("Truth Labels")

    #Classifier Prediction
    axs[2].bar(x=[0, 1, 2,3,4],height=counts) #Got bar heights based on value counts cell above. 
    #axs[2].set_ylabel("Event Count")
    axs[2].set_xlabel('Classifier Label')
    axs[2].set_xticks([0, 1, 2,3,4])
    axs[2].set_xticklabels(names,rotation=20)
    #axs[2].set_title("Classifier Predictions")

    #Classifier Score
    axs[3].hist(df['max_score_val'][:],bins=5,histtype='step',density=False)
    #axs[3].set_ylabel('Event Count')
    axs[3].set_xlabel('Classifier Score')
    #axs[3].set_title('Classifier Score')
    
    #fig.suptitle(f"{len(df)} Events from {df['run'][0]}",fontsize=15)
    fig.tight_layout(pad=3, w_pad=0.5, h_pad=1.0) 
    """
    runid = df['run'][0]
    return fig.savefig(f'characterization_plots_{runid}.png')
"""
Filter 2: 
- Make i3 from only events in csv.
"""
def events_cut(frame, event_csv):
    df = pd.read_csv(f'{event_csv}')
    uniform_events  = df['event'][:].values

    if frame['I3EventHeader'].sub_event_stream == 'NullSplit':
        return False
    elif frame['I3EventHeader'].sub_event_stream == 'InIceSplit':
        if frame['I3EventHeader'].event_id in uniform_events:
            return True
        else:
            return False

def filter2(infile, outdir,event_csv):
    drive, ipath =os.path.splitdrive(infile)
    path, ifn = os.path.split(ipath)
    infile_name = infile.split('/')[-1]
    tray = I3Tray()
    tray.Add('I3Reader', FilenameList=[infile])
    tray.AddModule(events_cut,event_csv = event_csv)
    tray.Add('I3Writer', 'EventWriter',
    FileName= outdir+'/uniform_energy_'+infile_name,
        Streams=[icetray.I3Frame.TrayInfo,
        icetray.I3Frame.Geometry,
        icetray.I3Frame.Calibration,
        icetray.I3Frame.DetectorStatus,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Physics, 
        icetray.I3Frame.Stream('S')],
        DropOrphanStreams=[icetray.I3Frame.DAQ]) 
    tray.AddSegment(I3HDFWriter, Output = f'{outdir}uniform_energy_{infile_name}.hd5', Keys = ['I3EventHeader','I3MCWeightDict',\
    'ml_suite_classification','NuGPrimary','PoleMuonLinefit', 'PoleMuonLinefitParams', 'PoleMuonLlhFitMuE', 'PoleMuonLlhFitFitParams',\
    'PoleMuonLlhFit','PolyplopiaInfo','PolyplopiaPrimary','I3MCTree','I3MCTree_preMuonProp','classification'], SubEventStreams=['InIceSplit'])
    tray.AddModule('TrashCan','can')

    tray.Execute()
    tray.Finish()

"""
Filter 3:
- Return Q frames only. 
- Split i3 file to be more managable for steamshovel to go through. 
- Maybe make another hdf just in case?
"""

def filter3(infile, run_id,dir):
    drive, ipath =os.path.splitdrive(infile)
    path, ifn = os.path.split(ipath)
    infile_name = infile.split('/')[-1]
    name_run = f'daq_{run_id}'
    #print(name_run)
    #new_daq_out = os.join(outdir,name_run)
    os.mkdir(f'daq_{run_id}')
    outdir = os.path.join(dir,name_run,"")
    #print(outdir)
    tray = I3Tray()
    tray.Add('I3Reader', FilenameList=[infile])
    tray.Add('I3MultiWriter', 'EventWriter',
    FileName= outdir+'daq_only-%04u_'+infile_name,
        Streams=[icetray.I3Frame.TrayInfo,
        icetray.I3Frame.Geometry,
        icetray.I3Frame.Calibration,
        icetray.I3Frame.DetectorStatus,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Stream('S')],
        SizeLimit = 2*10**6,)
    tray.AddModule('TrashCan','can')

    tray.Execute()
    tray.Finish()
"""
Make steamshovel videos
"""
movie_script = '/home/icecube/Desktop/eliz_zooniverse/icecubezooniverseproj_ver3/launch/Phase1/remake_steamshovel_movies_for_launch.py'
def get_steamshovel(indir,run_id):
    names = []
    compressed_daq = f'compressed_{run_id}'
    if os.path.isdir(compressed_daq) != True:
        os.mkdir(compressed_daq)
    else:
        pass
    for filename in os.listdir(indir):
        f = os.path.join(indir,filename)
        if filename.startswith('daq'):
            names.append(f)

            #subprocess.call(f"steamshovel {gcd} {f} --vanillaconsole --script {movie_script} --batch", shell = True)
            #note to self: can also try to add compression here
            #print(f"Finished Videos in: {filename}")
    names_sorted = sorted(names)
    print(f"List of files:\n{names_sorted}")
    for name in names_sorted:
        subprocess.call(f"steamshovel {gcd} {name} --vanillaconsole --script {movie_script} --batch", shell = True)
        print(f"Finished Videos in: {name}")

"""
Making manifest
"""
#After compressing videos using Handbrake...
def get_manifest(path, event_csv):
    #path : path to directory that holds compressed videos. 
    files = os.listdir(path) #insert folder path to compressed videos
    #return events that were actually made into videos. 
    structured_files = []
    for file in files:
        filename = file.split('_')
        event = int(filename[4])
        structured_files.append({'event':event})
    to_vids = pd.DataFrame.from_dict(structured_files)
    event_df = pd.read_csv(f'{event_csv}')
    event_df = event_df.drop(columns=['Unnamed: 0'], axis=1)
    k = event_df.merge(to_vids, left_on='event', right_on='event')

    #return file path and its respective events. 
    subj_path = []
    for f in glob.iglob(f'{path}/compressed_*.mp4',recursive=True):
        f_split = f[100:].split('_')
        event_split = int(f_split[4])
        subj_path.append({'filepath':f,'event':event_split})
    
    #make filepaths into dataframe
    s = pd.DataFrame.from_dict(subj_path)
    s1 = s.merge(k, left_on='event', right_on = 'event') #merge with events csv that only has events that made it through steamshovel.
    s2 = s1.set_index(s1['filepath'],inplace=False).T
    s3 = s2.drop(labels='filepath',axis=0)
    s3.to_csv('events_with_videos_manifest.csv')
    dict_manifest = s3.to_dict('dict')
    return dict_manifest
    
def make_subject_set(set_name,usr,pswd):
    Panoptes.connect(username=user, password=pswd)
    proj = Project.find('19023') #links to name that neutrino project
    #for workflow in proj.links.workflows:
        #print(workflow.display_name)
    subject_set = SubjectSet()

    subject_set.links.project = proj
    print(f"Subject Set Name: {set_name}")
    #set_name = input()
    subject_set.display_name = f"{set_name}"
 
    #subject_set.links.project = proj
    subject_set.save()
    new_set_id = subject_set.id
    print(f"Subject Set ID: {new_set_id}")
    proj.save()
    proj.reload()
    print(proj.links.subject_sets)


def uploader(path,event_csv,new_set_id,usr,pswd):
    #copy of get manifest bc return of func is not callable
    files = os.listdir(path) #insert folder path to compressed videos
    #return events that were actually made into videos. 
    structured_files = []
    for file in files:
        filename = file.split('_')
        event = int(filename[4])
        structured_files.append({'event':event})
    to_vids = pd.DataFrame.from_dict(structured_files)
    event_df = pd.read_csv(f'{event_csv}')
    event_df = event_df.drop(columns=['Unnamed: 0'], axis=1)
    k = event_df.merge(to_vids, left_on='event', right_on='event')

    #return file path and its respective events. 
    subj_path = []
    for f in glob.iglob(f'{path}/compressed_*.mp4',recursive=True):
        f_split = f[100:].split('_')
        event_split = int(f_split[4])
        subj_path.append({'filepath':f,'event':event_split})
    
    #make filepaths into dataframe
    s = pd.DataFrame.from_dict(subj_path)
    s1 = s.merge(k, left_on='event', right_on = 'event') #merge with events csv that only has events that made it through steamshovel.
    s2 = s1.set_index(s1['filepath'],inplace=False).T
    s3 = s2.drop(labels='filepath',axis=0)
    #s3.to_csv('events_with_videos_manifest.csv')
    dict_manifest = s3.to_dict('dict')


    Panoptes.connect(username=usr, password=pswd)
    proj = Project.find('19023') #links to name that neutrino project

    print(f"Confirm if subject set {new_set_id} is listed below.")
    
    proj.reload()
    print(proj.links.subject_sets)
    proj.reload()
    print(proj.links.subject_sets)

    new_subjects = []
    count = 0
    for filename, metadata in dict_manifest.items():
        subject = Subject()

        subject.links.project = proj
        subject.add_location(filename)

        subject.metadata.update(metadata)

        subject.save()
        new_subjects.append(subject)
        count +=1
        print(f'Number of times loop has ran: {count}')

    subject_set = SubjectSet.find(new_set_id)
    subject_set.add(new_subjects)
