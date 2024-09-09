from collections import Counter
import pickle

from Bio import pairwise2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd



#------------------------------------------------------------#
#-- Sequence alignment

# Quadratic mismatch function
# One semitone gives a mismatch of -1
def mismatch_fn(f1, f2):
    return - (abs(np.log2(float(f1)/float(f2))) * 12.)**2


def get_note_deviation(al1, al2, on1, on2, off1, off2):
    # Get positions in alignment without gaps
    not_gap1 = al1 != '-'
    not_gap2 = al2 != '-'

    # Find all positions where notes are matched
    idx = (not_gap1) & (not_gap2)
    
    # Calculate the frequency deviation in cents
    d_cents = 1200 * np.log2(al1[idx].astype(float) / al2[idx].astype(float))

    # Transform this into the relative indices of the on / off vectors
    i1 = (-1 + np.cumsum(not_gap1.astype(int)))[idx]
    i2 = (-1 + np.cumsum(not_gap2.astype(int)))[idx]

    # Calcualate on / off deviation
    d_on = on1[i1] - on2[i2]
    d_off = off1[i1] - off2[i2]

    return d_cents, d_on, d_off


def is_within_note(o1, on2, off2):
    return (o1 >= on2) & (o1 <= off2)


# Class 1: Serious disagreement
#          Onsets / offs do not overlap
# Class 2: Unison many-to-one (disagreement over number of notes)
#          Onsets / offs overlap, and frequency difference is within 100 cents
# Class 3: Non-unison many-to-one (disagreement over number of notes and also frequency)
#          Onsets / offs overlap, and frequency difference is greater than 100 cents
#          I imagine that this will correspond to disagreements over ornaments and passing notes
#
### For now, three classes is enough. It might be good create more detailed classes,
### and to split onset and frequency disagreements
def get_gap_class(freq, i_on, on_match_freq, i_off, off_match_freq, freq_cut=100):
    # If true, return Class 1:
    #       the note does not overlap with any note
    if (i_on == -1) and (i_off == -1):
        return 1

    # If true, then the note falls entirely within another note
    elif i_on == i_off:
        # If true, return Class 2:
        #       the frequencies of the matched notes are within a semitone
        if abs(1200 * np.log2(freq / on_match_freq)) <= freq_cut:
            return 2
        # Else, return Class 3:
        #       the frequencies of the matched notes are not within a semitone
        else:
            return 3

    # If true, then the note partly overlaps with (straddles) two other notes
    elif (i_on >= 0) & (i_off >= 0):
        # If true, return Class 2:
        #       the frequencies of one of the matched notes are within a semitone
        if abs(1200 * np.log2(freq / on_match_freq)) <= freq_cut:
            return 2
        # If true, return Class 2:
        #       the frequencies of one of the matched notes are within a semitone
        elif abs(1200 * np.log2(freq / off_match_freq)) <= freq_cut:
            return 2
        # Else, return Class 3:
        #       the frequencies of neither of the matched notes are within a semitone
        else:
            return 3


    # If true, then the note partly overlaps with another note
    elif (i_on >= 0) & (i_off == -1):
        # If true, return Class 2:
        #       the frequencies of the matched notes are within a semitone
        if abs(1200 * np.log2(freq / on_match_freq)) <= freq_cut:
            return 2
        # Else, return Class 3:
        #       the frequencies of the matched notes are not within a semitone
        else:
            return 3


    # If true, then the note partly overlaps with another note
    elif (i_on == -1) & (i_off >= 0):
        # If true, return Class 2:
        #       the frequencies of the matched notes are within a semitone
        if abs(1200 * np.log2(freq / off_match_freq)) <= freq_cut:
            return 2
        # Else, return Class 3:
        #       the frequencies of the matched notes are not within a semitone
        else:
            return 3


def find_last_note_before_gap(al, i0):
    idx = np.where(al[:i0]!='-')[0]
    if len(idx):
        return float(al[idx[-1]])
    else:
        return np.nan


def classify_gaps(key, al1, al2, on1, on2, off1, off2):
    # Get positions in alignment with gaps
    is_gap1 = al1 == '-'
    is_gap2 = al2 == '-'
    # Get indices of non-gaps, for aligning ons / offs
    i1 = (-1 + np.cumsum((is_gap1==0).astype(int)))
    i2 = (-1 + np.cumsum((is_gap2==0).astype(int)))

    out = []

    # The gap in sequence 1 corresponds to an extra note in sequence 2.
    # Find out if the on of this extra note falls within
    # a note in sequnece 1.
    # Then do the same for the off of the extra note.
    for i in np.where(is_gap1)[0]:
        # Check whether this note onset from sequence 2 falls within
        # any pair of onset and offset from sequence 1
        on_within_note = is_within_note(on2[i2[i]], on1, off1)

        # If the note onset is within at least one pair,
        # record the index and frequency of the first one
        if np.sum(on_within_note):
            i_on = np.where(on_within_note)[0][0]
            on_match_freq = float(al1[is_gap1==0][i_on])
        else:
            i_on = -1
            on_match_freq = np.nan


        # Check whether this note offset from sequence 2 falls within
        # any pair of onset and offset from sequence 1
        off_within_note = is_within_note(off2[i2[i]], on1, off1)

        # If the note offset is within at least one pair,
        # record the index and frequency of the first one
        if np.sum(off_within_note):
            i_off = np.where(off_within_note)[0][0]
            off_match_freq = float(al1[is_gap1==0][i_off])
        else:
            i_off = -1
            off_match_freq = np.nan

        freq = float(al2[i])
        gap_class = get_gap_class(freq, i_on, on_match_freq, i_off, off_match_freq)
        out.append([key, i, freq, i_on, on_match_freq, i_off, off_match_freq, gap_class])

    return out
    

# Convert frequency float to interval, then to string, then to list (formatting requirements for downstream code...)
# Then get the top alignment (if there is more than one alignment with
# the same score, then just take the first one)
def compare_note_sequences(on1, seq1, off1, on2, seq2, off2, gap_pen=-0.25):
    seq1 = list(seq1.astype(int).astype(str))
    seq2 = list(seq2.astype(int).astype(str))
    al1, al2, score, _, al_len = pairwise2.align.globalcs(seq1, seq2, mismatch_fn, gap_pen, gap_pen, gap_char=['-'], one_alignment_only=1)[0]
    al1, al2 = np.array(al1), np.array(al2)

    d_cents, d_on, d_off = get_note_deviation(al1, al2, on1, on2, off1, off2)
    gap_data =      classify_gaps(0, al1, al2, on1, on2, off1, off2)
    gap_data.extend(classify_gaps(1, al2, al1, on2, on1, off2, off1))

    return al1, al2, score, al_len, d_cents, d_on, d_off, np.array(gap_data)



def compare_symbolic_sequences(seq1, seq2, gap_pen=-0.25):
    seq1 = list(seq1.astype(int).astype(str))
    seq2 = list(seq2.astype(int).astype(str))
    al1, al2, score, _, al_len = pairwise2.align.globalcs(seq1, seq2, mismatch_fn, gap_pen, gap_pen, gap_char=['-'], one_alignment_only=1)[0]
    al1, al2 = np.array(al1), np.array(al2)
    return al1, al2, score, al_len


# If non-gap matches are within 1 semitone, accept it as correct
def get_PID(al1, al2, threshold=1):
    score = 0
    for a1, a2 in zip(al1, al2):
        if (a1 != '-') & (a2 != '-'):
            if (-mismatch_fn(a1, a2) < threshold):
                score += 1
    return 100 * score / len(al1)





if __name__ == "__main__":

    pass







