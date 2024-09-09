from collections import Counter, defaultdict
import pickle

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp, pearsonr, spearmanr

import plots


### Finds all pairs notes that overlap,
### returns the pairs, along with the fraction overlap
def notes_overlap_all(data, t1=0, t2=1):
    on1, off1, dur1 = [data[t1][k] for k in ['note_on', 'note_off', 'note_dur']]
    on2, off2, dur2 = [data[t2][k] for k in ['note_on', 'note_off', 'note_dur']]

    # Expand onsets and offsets onto a 2d mesh
    # (1 dimension per transcription)
    s1, s2 = np.meshgrid(on2, on1)
    e1, e2 = np.meshgrid(off2, off1)

    # Find notes that fully overlap with another note
    i0, j0 = np.where((s1 >= s2) & (e1 <= e2) | (s1 <= s2) & (e1 >= e2))

    # Find notes that partially overlap with another note
    i1, j1 = np.where((s1 >= s2) & (s1 <= e2) | (e1 >= s2) & (e1 <= e2))

    used = set()
    df = pd.DataFrame(columns=['i', 'j', 'ol_i', 'ol_j'])
    for i, j in zip(*[np.append(i0, i1), np.append(j0, j1)]):
        # Avoid duplicates (which can happen if segments exactly match)
        if (i, j) in used:
            continue
        used.add((i,j))

        overlap = min(off1[i], off2[j]) - max(on1[i], on2[j])
        ol_i = overlap / dur1[i]
        ol_j = overlap / dur2[j]
        df.loc[len(df)] = [i, j, ol_i, ol_j]

    df['i'] = df['i'].astype(int)
    df['j'] = df['j'].astype(int)

    return df


### Not used?
def match_notes(df, ni, nj):
    match_i = [[] for i in range(ni)]
    match_j = [[] for i in range(nj)]

    overlap_i = np.zeros(ni)
    overlap_j = np.zeros(nj)

    for i, j, oi, oj in zip(df.i, df.j, df.ol_i, df.ol_j):
        match_i[i].append(j)
        match_j[j].append(i)
        overlap_i[i] += oi
        overlap_j[j] += oj

    return match_i, match_j, overlap_i, overlap_j


### Takes the output of "notes_overlap_all" as input,
### and groups notes if at least one of a pair of notes
### has a fractional overlap >= "cut".
### If many notes are connected via chains of overalapping notes, 
### they are assigned to one group
def group_notes(df, cut=0.5):
    df = df.sort_values(by=['i', 'j'])
    pairs = df.loc[(df.ol_i>=cut) | (df.ol_j>=cut), ['i', 'j']].values

    groups = {}
    used1, used2 = {}, {}
    for i, j in zip(*pairs.T):
        # If "i" has been added to a group, then add "j" to the same group
        if i in used1:
            g = used1[i]
        # If "j" has been added to a group, then add "i" to the same group
        elif j in used2:
            g = used2[j]
        # Else create a new group for "i" and "j"
        else:
            g = len(groups)
            groups[g] = [set(), set()]

        # Add group to the keys for "i" and "j"
        used1[i] = g
        used2[j] = g

        # Add "i" and "j" to the correct group
        groups[g][0].add(i)
        groups[g][1].add(j)

    return groups


def annotate_mismatches(data, t1=0, t2=1):
    groups = group_notes(notes_overlap_all(data))
    X1 = np.array([data[t1][k] for k in ['note_dur', 'f_median_diff', 'c_std', 'c_grad']]).T
    X2 = np.array([data[t2][k] for k in ['note_dur', 'f_median_diff', 'c_std', 'c_grad']]).T
    match, mismatch1, mismatch2 = [], [], []
    for sets in groups.values():
        if (len(sets[0]) == 1) & (len(sets[1]) == 1):
            # Only add a single note
            for i in sets[0]:
                match.append(X1[i])
            for j in sets[1]:
                match.append(X2[j])
        elif len(sets[0]) < len(sets[1]):
            # Only add the longer note(s)
            for i in sets[0]:
                mismatch1.append(X1[i])
            for j in sets[1]:
                mismatch2.append(X2[j])
        elif len(sets[0]) > len(sets[1]):
            # Only add the longer note(s)
            for i in sets[0]:
                mismatch2.append(X1[i])
            for j in sets[1]:
                mismatch1.append(X2[j])
    return [np.array(x) for x in [match, mismatch1, mismatch2]]


def get_all_match_mismatch(data):
    match, mismatch1, mismatch2 = [], [], []
    for d in data:
        a, b, c = annotate_mismatches(d)
        match.extend(list(a))
        mismatch1.extend(list(b))
        mismatch2.extend(list(c))
    return [np.array(x) for x in [match, mismatch1, mismatch2]]


### For each pair of transcriptions, calculate whether the distributions of
### ["note_dur" : note duration,
###  "f_median_diff": difference in median pitch of note and the assigned note pitch,
###  "c_std": standard deviation of pitches (in cents) within note,
###  "c_grad": gradient of linear fit to pitches (in cents) within note]
### are significantly different. Uses a Mann-Whitney U test
def group_differences(data, groups={}, t1=0, t2=1):
    if not len(groups):
        groups = group_notes(notes_overlap_all(data, t1=t1, t2=t2))
    xcat = ['note_dur', 'amp_std', 'c_std', 'c_grad']
    mwu = []
    for i0, x in enumerate(xcat):
        # Make sure the data is actually there (amp_std is missing for many)
        if (x not in data[t1]) or (x not in data[t2]):
            mwu.append([[np.nan]*2]*3)
            continue
        # Perfectly matched notes
        diff1 = [y for g in groups.values() for y in [data[t1][x][list(g[0])[0]], data[t2][x][list(g[1])[0]]] if len(g[0])==1 and len(g[1])==1]
        # Many-to-many matches
        diff2, diff3, diff4 = [], [], []
        for g in groups.values():
            if len(g[0]) > len(g[1]):
                diff2.extend([data[t2][x][j] for j in g[1]])
                diff3.extend([data[t1][x][i] for i in g[0]])
            elif len(g[0]) < len(g[1]):
                diff2.extend([data[t1][x][i] for i in g[0]])
                diff3.extend([data[t2][x][j] for j in g[1]])
            elif (len(g[0]) == len(g[1])) & (len(g[0]) > 1):
                diff4.extend([data[t1][x][i] for i in g[0]])
                diff4.extend([data[t2][x][j] for j in g[1]])

        diff1, diff2, diff3, diff4 = [np.abs(d) for d in [diff1, diff2, diff3, diff4]]
        mwu.append([mannwhitneyu(diff1, diff2),
                    mannwhitneyu(diff1, diff3),
                    mannwhitneyu(diff1, diff4)])
    return np.array(mwu)


def get_group_diff(df, data):
    pdiff = []
    for i in df.loc[df.Precision.notnull()].index:
        pdiff.append(group_differences(data[i], t1=df.loc[i, "t1"], t2=df.loc[i, "t2"]))
    return np.array(pdiff)


### For each pair of transcriptions, calculate the mean values of
### ["note_dur" : note duration,
###  "f_median_diff": difference in median pitch of note and the assigned note pitch,
###  "c_std": standard deviation of pitches (in cents) within note,
###  "c_grad": gradient of linear fit to pitches (in cents) within note]
### Calculate separate means for matched notes, and mismatched notes
def mean_group_differences(data, groups={}, t1=0, t2=1):
    if not len(groups):
        groups = group_notes(notes_overlap_all(data, t1=t1, t2=t2))
    xcat = ['note_dur', 'amp_std', 'c_std', 'c_grad']
    mean = []
    for i0, x in enumerate(xcat):
        # Make sure the data is actually there (amp_std is missing for many)
        if (x not in data[t1]) or (x not in data[t2]):
            mean.append([np.nan]*4)
            continue
        # Perfectly matched notes
        diff1 = [y for g in groups.values() for y in [data[t1][x][list(g[0])[0]], data[t2][x][list(g[1])[0]]] if len(g[0])==1 and len(g[1])==1]
        # Many-to-many matches
        diff2, diff3, diff4 = [], [], []
        for g in groups.values():
            if len(g[0]) > len(g[1]):
                diff2.extend([data[t2][x][j] for j in g[1]])
                diff3.extend([data[t1][x][i] for i in g[0]])
            elif len(g[0]) < len(g[1]):
                diff2.extend([data[t1][x][i] for i in g[0]])
                diff3.extend([data[t2][x][j] for j in g[1]])
            elif (len(g[0]) == len(g[1])) & (len(g[0]) > 1):
                diff4.extend([data[t1][x][i] for i in g[0]])
                diff4.extend([data[t2][x][j] for j in g[1]])

        diff1, diff2, diff3, diff4 = [np.abs(d) for d in [diff1, diff2, diff3, diff4]]
        mean.append([np.mean(d) for d in [diff1, diff2, diff3, diff4]])
    return np.array(mean)


def analyse_transcription(data, plot=False, t1=0, t2=1):
    ol = notes_overlap_all(data, t1=t1, t2=t2)
    ni = len(data[t1]['note_on'])
    nj = len(data[t2]['note_on'])
    match_i, match_j, overlap_i, overlap_j = match_notes(ol, ni, nj)
    groups = group_notes(ol)

    match = np.sum([(len(x)==1)&(len(y)==1) for x, y in groups.values()]) * 2
    no_match = np.sum([len(x)==0 for x in match_i + match_j])
    mismatch = np.sum([len(x) + len(y) for x, y in groups.values() if (len(x)!=1)|(len(y)!=1)])

    print(f"{data['song']}\nMatched notes = {match}")
    print(f"Unmatched notes = {no_match}")
    print(f"Mis-matched notes = {mismatch}")

    if plot:
#       plots.plot_alignment(data)
        plots.group_differences(data, groups)

    ave_mismatch = np.mean([len(x) + len(y) for x, y in groups.values() if (len(x)!=1)|(len(y)!=1)])

    return match, no_match, mismatch, ave_mismatch


def update_df(df, data):
    col = ['match', 'nomatch', 'mismatch', 'ave_mismatch']
    for i in df.loc[df.Precision.notnull()].index:
         t1, t2 = df.loc[i, ["t1", "t2"]]
         df.loc[i, col] = analyse_transcription(data[i], t1=t1, t2=t2)
         df.loc[i, 'mean_dur'] = np.nanmean([x for j in [t1, t2] for x in data[i][j]['note_dur']])
         df.loc[i, 'mean_std'] = np.nanmean([x for j in [t1, t2] for x in data[i][j]['c_std']])
    return df


def update_df_2(df0, pdiff):
    xcat = ['note_dur', 'f_median_diff', 'c_std', 'c_grad']
    for i, x in enumerate(xcat):
        if i == 1:
            continue
        df0[f"p_{x}_big"] = pdiff[:,i,0,1].round(3)
        df0[f"p_{x}_small"] = pdiff[:,i,1,1].round(3)
    return df0
        

### Were note pitches manually altered by transcribers?
### We check this by comparing the median of the pitches
### that fall within the note onset and offset,
### and compare this with the note pitch assigned by the transcriber
def pitch_alteration(data):
    used = set()
    f_med_diff = defaultdict(list)
    note_dur = defaultdict(list)
    for d in data:
        song = d['song']
        for k, v in d.items():
            if not isinstance(k, int):
                continue
            t = v['transcriber']
            if (t, song) in used:
                continue
            used.add((t, song))
            f_med_diff[t].extend(list(v['c_median_diff']))
            note_dur[t].extend(list(v['note_dur']))
    f_med_diff = {k: np.array(v) for k, v in f_med_diff.items()}
    note_dur = {k: np.array(v) for k, v in note_dur.items()}
    return f_med_diff, note_dur
                

### Need a function that determines whether a note's pitch has been changed manually.
### Compare the note freq with the median value of pitches (or whatever TONY uses to assign note pitch)


### Need an analysis to quantify pitch stability of a note,
### and to see whether it differs from on/offset matches vs disagreements


### What happens in disagreements when there are no clear differences in pitch?
### Is there a lyrical component that we can't capture with algorithms?
### Can we find such signatures in syllabic onsets by looking at amplitude fluctuations?

### Are there systematic differences between transcribers within songs?
### i.e. one transcriber tends to annotate ornaments more as separate notes?
### Are these systematic differences apparent across songs?


def get_lr(X, Y, p=False, nozero=False, xlog=False, ylog=False, spear=False):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    
    try:
        if xlog:
            X = np.log(X)

        if ylog:
            Y = np.log(Y)

        if nozero:
            idx = (np.isfinite(X)) & (np.isfinite(Y)) & (X > 0) & (Y > 0)
        else:
            idx = (np.isfinite(X)) & (np.isfinite(Y))

        if spear:
            corr_fn = spearmanr
        else:
            corr_fn = pearsonr

        if p:
            return corr_fn(X[idx], Y[idx])
        else:
            return corr_fn(X[idx], Y[idx])[0]
    except Exception as e:
        if p:
            return [np.nan, np.nan]
        else:
            return np.nan


