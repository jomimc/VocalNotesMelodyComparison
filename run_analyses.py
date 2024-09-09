from collections import defaultdict
from itertools import product
import os
from pathlib import Path
import pickle
from subprocess import run
import sys
import time

import mir_eval
import numpy as np
import pandas as pd
from scipy.stats import linregress

import audio_analysis as AA
import compare_transcriptions as CT
import matching_notes as MN
from vn_io import PATH_RES 
import vn_io


### Main function to transcribe a recording using an algorithm, and compare with the ground truth
def evaluate_algorithm_single(data, t1=0, t2=1, redo=False):
    ts = time.time()
    try:
#       base = PATH_RES.joinpath(alg)
#       base.mkdir(parents=True, exist_ok=True)

        # For transcriptions with accurate note durations
        # Evaluate transcription using standard MIR metrics
        ref_intervals = np.array([data[t1]['note_on'], data[t1]['note_off']]).T
        ref_pitches = data[t1]['note_freq']

        est_intervals = np.array([data[t2]['note_on'], data[t2]['note_off']]).T
        est_pitches = data[t2]['note_freq']

        try:
            res = mir_eval.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)
        except Exception as e:
            print(f"Error analyzing {data[t1]['path']}\n\t{e}")
            res = {}

        # Evaluate transcription using metrics that can be
        # clearly interpreted by musicians
        inputs = [data[t][k] for t in [t1, t2] for k in ['note_on', 'note_freq', 'note_off']]
        al1, al2, score, al_len, d_cents, d_on, d_off, gap_data = CT.compare_note_sequences(*inputs)
        res['PID'] = CT.get_PID(al1, al2, 1)
        res['alignment_score'] = score
        res['mean_cents_diff'] = np.mean(np.abs(d_cents))
        res['mean_onset_diff'] = np.mean(np.abs(d_on))
        res['mean_offset_diff'] = np.mean(np.abs(d_off))

        res['non_overlap_notes'] = np.sum(gap_data[:,7] == 1)
        res['missing_notes'] = np.sum((gap_data[:,7] == 1) & (gap_data[:,0]==0))
        res['extra_notes'] = np.sum((gap_data[:,7] == 1) & (gap_data[:,0]==1))
        res['overlap_unison_notes'] = np.sum(gap_data[:,7] == 2)
        res['overlap_nonunison_notes'] = np.sum(gap_data[:,7] == 3)

        res['n_seg_t1'] = len(data[t1]['note_on'])
        res['n_seg_t2'] = len(data[t2]['note_on'])

        # Save results
#       for c in ['ID', 'corpus', 'name']:
#           res.update({c: data[c]})
#       pickle.dump(res, open(path, 'wb'))

        return res
    except Exception as e:
#       print(data['ID'], data['name'], alg, e)
        print(f"Error analyzing {data[t1]['path']}\n\t{e}")
        return {}


def analyse_note_pitch(data):
    for t, d in data.items():
        if not isinstance(t, int):
            continue
        T, F = [d['pitch'][k] for k in ['time', 'freq']]
        C = np.log2(F) * 1200
        med, std, grad = [], [], []
        on, off, freq = [d[k] for k in ['note_on', 'note_off', 'note_freq']]
        for i in range(len(on)):
            idx = (T>=on[i])&(T<=off[i])
            if np.sum(idx) == 0:
                med.append(np.nan)
                std.append(np.nan)
                grad.append(np.nan)
            else:
                med.append(np.median(F[idx]))
                std.append(np.std(C[idx]))
                grad.append(linregress(T[idx], C[idx])[0])
        data[t]['f_median'] = np.array(med)
        data[t]['c_std'] = np.array(std)
        data[t]['c_grad'] = np.array(grad)
        data[t]['f_median_diff'] = data[t]['f_median'] - data[t]['note_freq']
        data[t]['c_median_diff'] = np.log2(data[t]['f_median'] / data[t]['note_freq'])*1200
    return data


def update_note_amplitude(data):
    T, A = data['amplitude']
    for t, d in data.items():
        if not isinstance(t, int):
            continue
        amp_std = []
        on, off, freq = [d[k] for k in ['note_on', 'note_off', 'note_freq']]
        for i in range(len(on)):
            idx = (T>=on[i])&(T<=off[i])
            if np.sum(idx) == 0:
                amp_std.append(np.nan)
            else:
                amp_std.append(np.std(np.log10(A[idx])))
        data[t]['amp_std'] = amp_std
    return data


def evaluate_algorithm_all():
    res, data = [], []
    groups = ['Russian', 'Japan', 'China', 'Alpine', 'Jewish']
#   groups = ['Russian']#'Japan', 'China', 'Alpine', 'Jewish']
    divider = ['__', '__', '_', '__', '__']
    for g, d in zip(groups, divider):
        base = vn_io.PATH_DATA.joinpath(g, "Analysis")
        path_dict = vn_io.get_path_list(base, divider=d)
        for song, paths in path_dict.items():
            # Load the transcriptions ("notes", or "segments")
            if g == 'Alpine':
                if not vn_io.all_files_there(paths, m=2):
                    print(f"Segments files missing for {song}")
                    continue
                trans = vn_io.load_transcriptions(paths, 'segments')
            else:
                if not vn_io.all_files_there(paths, m=3):
                    print(f"Notes missing for {song}")
                    continue
                trans = vn_io.load_transcriptions(paths)

            # Add the song name
            trans['song'] = song

            # Add the pitch trace
            try:
                for i, p in enumerate(paths.values()):
                    trans[i]['pitch'] = vn_io.load_pitches(p['pitches'])

                # Analyse the pitch content of each note
                trans = analyse_note_pitch(trans)
            except Exception as e:
                    print(f"Pitch files missing for {song}\n\t{e}")
                    continue

            # Add the amplitude data
            path = vn_io.match_audio(g, song)
            if not isinstance(path, type(None)):
                try:
                    trans['amplitude'] = AA.signal_energy(*AA.load_audio(path))
                    trans = update_note_amplitude(trans)
                except Exception as e:
                    print(f"{e}\t{g}\t{song}")
                
            n_trans = len(paths)
            for i, j in product(range(n_trans), range(n_trans)):
                if i >= j:
                    continue
                r = evaluate_algorithm_single(trans, t1=i, t2=j)
                r['song'] = song
                r['group'] = g
                r['t1'] = i
                r['t2'] = j
                r['t1_name'] = trans[i]['transcriber']
                r['t2_name'] = trans[j]['transcriber']
                res.append(r)
                data.append(trans)
    df = pd.DataFrame(data=res)
    df['n_seg_diff'] = df.n_seg_t1 - df.n_seg_t2
    return df, data


def run_all():
    df, data = evaluate_algorithm_all()
    df = MN.update_df(df, data)

    pdiff = MN.get_group_diff(df, data)
    mean_diff = np.array([MN.mean_group_differences(data[i], t1=df.loc[i, "t1"], t2=df.loc[i, "t2"]) for i in df.loc[df.Precision.notnull()].index])

    df0 = df.loc[df.Precision.notnull(), df.columns[24:]]
    df0 = MN.update_df_2(df0, pdiff)
    df0['trans_team'] = [f"{x}_{y}" for x, y in zip(df0.t1_name, df0.t2_name)]
    return df, df0, data, pdiff, mean_diff



