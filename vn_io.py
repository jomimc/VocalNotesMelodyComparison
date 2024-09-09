from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import pandas as pd

# path to ANALYSIS folder
PATH_BASE = Path("/home/jmcbride/projects/VocalNotes")
PATH_DATA = PATH_BASE.joinpath("Data")
PATH_RES  = PATH_BASE.joinpath("Results")

BASE = PATH_DATA.joinpath("Japan", "Analysis")


#ATH_TO_ANALYSIS = r"C:\Users\Administrator\Documents\Google_Drive\BirdSongSpeech\Alpine\Analysis"
#ASE = Path(PATH_TO_ANALYSIS)

# Filename for file with the list of transcriber initials
PATH_TRANS_NAME = "transcribers.csv"

# Filename for file with the list of songs / recordings
PATH_SONG_NAME = "recordings.csv"

# The type of divider used in filenames to separate SONG, TRANSCRIBER, and FILETYPE:
# e.g. "SONG__TRANSCRIBER__FILETYPE"
DIVIDER = "__" 
    
FILETYPE_LIST = ['notes', 'segments', 'pitches']


# Checks whether the list of transcribers in PATH_TRANS_NAME matches the
# list of directories in a song directory.
# Identifies both missing directories, and incorrectly labelled directories
def check_transcriber_directories(song, transcriber_dirs, transcriber_list):
    missing = []
    wrong = []
    
    for t in transcriber_list:
        if t not in transcriber_dirs:
            missing.append(t)
            
    for t in transcriber_dirs:
        if t not in transcriber_list:
            wrong.append(t)
    
    if len(missing):
        print(f"No data for Transcribers {missing}, for {song}\n")
        
    if len(wrong):
        print(f"Incorrect transcriber folder(s) {wrong}, for {song}\n")

    return missing + wrong

# Check if all files exist
# Only check existing directories "transcriber_dirs", rather than
# printing file_not_found errors for every file in "transcriber_list"
def check_all_files_exist(song, transcriber_dirs, base=BASE, divider=DIVIDER):
    file_extensions = ['.csv', '.svl']
    path_list = []
    csv_path_dict = defaultdict(dict)
    for td in transcriber_dirs:
        for ft in FILETYPE_LIST:
            for ext in file_extensions:
                path = base.joinpath(song, td, f"{song}{divider}{td}{divider}{ft}{ext}")
                if not path.exists():
                    print(f"File not found:\n\t{path}\n")
#                   print(f"{path}")
                else:
                    path_list.append(path)
                    if ext == '.csv':
                        csv_path_dict[td][ft] = path
    return path_list, csv_path_dict
    
    
# If a note is shorter than 10 ms, identify it as a potential error
def identify_short_notes(path, min_duration=0.01):
    try:
        df = pd.read_csv(path)
        if 'DURATION' not in df.columns:
            print(f"Incorrect column name in {path}\n\tPlease ensure that note duration column is called DURATION\n")
        duration = df['DURATION'].values 
    except:
        duration = np.loadtxt(path, usecols=(0,1,2), delimiter=',')[:,2]

    for i, dur in enumerate(duration):
        if dur < min_duration:
            print(f"Short duration found for note {i+1} in {path}\n\tPlease check if this is correct!\n")
        

# Try to read csv files, and print any errors that occur
# Not sure what to do with the ".svl" files
def check_file_contents(path):
    if path.suffix == '.csv':
        try:
            if 'notes' in path.stem:
                identify_short_notes(path)
            else:
                df = pd.read_csv(path)
        except Exception as e:
            print(f"Error found while reading {path}\n\t{e}\n")
            
            
# If durations of 'notes' or 'pitches' files are not within 20%
# of each other, identify it as a potential error
def check_duration_consistency(song, csv_path_dict, transcriber_dirs, threshold=0.2):
    for ft in FILETYPE_LIST:
        if ft == 'segments':
            continue
        total_duration = []
        for td in transcriber_dirs:
            if ft not in csv_path_dict[td]:
                continue
                
            path = csv_path_dict[td][ft]
            try:
                df = pd.read_csv(path)
            except:
                continue
                
            if 'DURATION' not in df.columns:
#               print(f"Incorrect column name in {path}\n\tPlease ensure that note duration column is called DURATION\n")
                continue
                
            if ft == 'notes':
                total_duration.append(df['DURATION'].sum())
            elif ft == 'pitches':
                total_duration.append(len(df))
                
        if len(total_duration) <= 1:
            continue
            
        mean_duration = sum(total_duration) / len(total_duration)
        if any([abs(1 - dur / mean_duration) > threshold for dur in total_duration]):
            print(f"Transcribers differ in total notated duration of the {ft} file for {song}\n")
            
            
    
# Run a series of checks for each song
def check_each_song(song, transcriber_list, base=BASE, divider=DIVIDER):
    song_dir = base.joinpath(song)
    if not song_dir.exists():
        print(f"Directory for recording {song} could not be found at\n\t{song_dir}\n")
        return
    
    transcriber_dirs = next(os.walk(song_dir))[1]
    check_transcriber_directories(song, transcriber_dirs, transcriber_list)
    path_list, csv_path_dict = check_all_files_exist(song, transcriber_dirs, base=base, divider=divider)
    
    if len(transcriber_dirs) > 1:
        check_duration_consistency(song, csv_path_dict, transcriber_dirs)
    
    for path in path_list:
        check_file_contents(path)
        
    
def check_all_files(base=BASE, divider=DIVIDER):
    song_list = pd.read_csv(base.joinpath(PATH_SONG_NAME))['name']
    transcriber_list = pd.read_csv(base.joinpath(PATH_TRANS_NAME))['initials'].values
    for song in song_list:
        check_each_song(song, transcriber_list, base=base, divider=divider)


def get_path_list(base=BASE, divider=DIVIDER):
    song_list = pd.read_csv(base.joinpath(PATH_SONG_NAME))['name']
    transcriber_list = pd.read_csv(base.joinpath(PATH_TRANS_NAME))['initials'].values
    path_dict = {}
    for song in song_list:
        song_dir = base.joinpath(song)
        transcriber_dirs = next(os.walk(song_dir))[1]
#       exclude = check_transcriber_directories(song, transcriber_dirs, transcriber_list)
#       if len(exclude):
#           print("Do something!")
        path_list, csv_path_dict = check_all_files_exist(song, transcriber_dirs, base=base, divider=divider)
        path_dict[song] = csv_path_dict
    return path_dict


def get_all_path_dict():
    groups = ['Russian', 'Japan', 'China', 'Alpine']
    for g in groups:
        base = PATH_DATA.joinpath(g, "Analysis")
        path_dict = get_path_list(base)


def all_files_there(path_dict, n=2, m=3):
    if len(path_dict) < n:
        return False
    if not np.all([len(d) == m for d in path_dict.values()]):
        return False
    return True
    

def load_notes(path):
    try:
        on, freq, dur = np.loadtxt(path, usecols=(0,1,2), delimiter=',').T
    except:
        on, freq, dur = np.loadtxt(path, usecols=(0,1,2), delimiter=',', skiprows=1).T
    cents = np.log2(freq / 440) * 1200
    return {'note_on':on, 'note_freq': freq, 'note_off':on + dur, 'note_dur':dur, 'note_cents':cents}


def load_pitches(path):
    try:
        tm, freq = np.loadtxt(path, usecols=(0,1), delimiter=',').T
    except:
        tm, freq = np.loadtxt(path, usecols=(0,1), delimiter=',', skiprows=1).T
    idx = freq > 0
    return {'time': tm[idx], 'freq':freq[idx]}


def load_transcriptions(path_dict, fn='notes'):
    out = {}
    for i, (transcriber, pd) in enumerate(path_dict.items()):
        # pd contains {'notes': path_to_notes, 'pitches':path_to_pitches, 'segments':path_to_segments}
        path = pd[fn]
        out[i] = load_notes(path)
        out[i]['path'] = path
        out[i]['transcriber'] = transcriber
    return out


def match_audio(group, song):
    path = list(PATH_DATA.joinpath(group, "Originals").glob(f"{song}*"))
    if len(path) == 1:
        return path[0]
    elif len(path) > 1:
        print(f"More than one recording matches?\n{group}\t{song}")
    else:
        print(f"No matching recording found\n{group}\t{song}")
    


def run_all_file_checks():
    res = []
    groups = ['Russian', 'Japan', 'China', 'Alpine', 'Jewish']
    divider = ['__', '__', '_', '__', '__']
    for g, d in zip(groups, divider):
        base = PATH_DATA.joinpath(g, "Analysis")
        path_dict = get_path_list(base, d)
        check_all_files(base, d)




