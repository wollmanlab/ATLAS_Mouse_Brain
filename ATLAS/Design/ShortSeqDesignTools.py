"""
ShortSeqDesignTools is a library of function used to design short sequences (primers, readout probes, encoder libraries, HCRs)
"""

import more_itertools as mit
import itertools as it
import numpy as np
import random
import Levenshtein 
import re 
import matplotlib.pyplot as plt
import seaborn as sns
import nupack
import time
from IPython.display import HTML
from ATLAS.Utils.nupacku import *
from ATLAS.Utils.sequ import *
from Bio.Seq import Seq

def get_rcseq(seq):
    """reverse complement

    Args:
        seq (str): _description_

    Returns:
        str: _description_
    """
    return str(Seq(seq).reverse_complement())


def lev_mat(seq_list,self_dist = 0):
    """ Distance functions based on levenshtein (lev) edit distance
    Lev counts the number of edits needed between seq (del, mut, inserts) to make them the same
    It returns the same distance as local seq alignment with specific weights. 
    It is pretty fast and good as a first pass screening tool to compare sequences. 

    inputs: 
    -------
    seq_list - list of seq (ATCG list)
    self_dist - value to place on diagonal (default 0)

    output: 
    -------
    symetrical distance matrix NxN (for N sequences), with self_dist on diagonal and the distances elsewhere.  
    """
    ij = list(it.combinations(range(len(seq_list)),2))
    lev_dist_mat = np.zeros((len(seq_list),len(seq_list)))
    for i in range(len(ij)):
        lev_dist_mat[ij[i][0],ij[i][1]]=Levenshtein.distance(seq_list[ij[i][0]],seq_list[ij[i][1]])
        lev_dist_mat[ij[i][1],ij[i][0]]=lev_dist_mat[ij[i][0],ij[i][1]]
    # add the diagonal of self_dist
    for i in range(len(seq_list)):
        lev_dist_mat[i,i]=self_dist
    return(lev_dist_mat)

def lev_mat2(seq_list1,seq_list2):
    """
    same as lev_mat buyt calcualted between two sets of distances, will return matrix MxN
    """
    ij = list(it.product(range(len(seq_list1)),range(len(seq_list2))))
    lev_dist_mat = np.zeros((len(seq_list1),len(seq_list2)))
    for i in range(len(ij)):
        lev_dist_mat[ij[i][0],ij[i][1]]=Levenshtein.distance(seq_list1[ij[i][0]],seq_list2[ij[i][1]])
    return(lev_dist_mat)
    
def binding_dist_oligo_mat(oligo_list_1,oligo_list_2 = None,hyb_conditions = None,return_matrices = False):
    """
    calculate binding properties for all pairing of oligos in list given nupacl hyb condition model.
    uses binding_dist_oligos_lists for actual calculation

    inputs: 
    -------
    oligo_list_1 : list of oligos
    oligo_list_2 : list of oligos (could be empty and them doing oligo_list_1 vs itself)
    
    """
    if oligo_list_2 is None:
        oligo_list_1 = np.array(oligo_list_1)
        oligo_pairs_ix = np.array([(i, j) for i, j in it.product(range(len(oligo_list_1)), repeat=2) if i <= j])
        oligo_pairs_1 = oligo_list_1[oligo_pairs_ix[:,0]]
        oligo_pairs_2 = oligo_list_1[oligo_pairs_ix[:,1]]
    else: 
        oligo_list_1 = np.array(oligo_list_1)
        oligo_list_2 = np.array(oligo_list_2)
        oligo_pairs_ix = np.array(list(it.product(range(len(oligo_list_1)),range(len(oligo_list_2)))))
        oligo_pairs_1 = oligo_list_1[oligo_pairs_ix[:,0]]
        oligo_pairs_2 = oligo_list_2[oligo_pairs_ix[:,1]]
    bps_list,dG_list = binding_dist_oilgo_lists(oligo_pairs_1,oligo_pairs_2,hyb_conditions)
    if return_matrices:
        if oligo_list_2 is None: 
            dG_mat = np.zeros((len(oligo_list_1),len(oligo_list_1)))
        else: 
            dG_mat = np.zeros((len(oligo_list_1),len(oligo_list_2)))
        bps_mat = np.zeros_like(dG_mat)
        for i in range(oligo_pairs_ix.shape[0]):
            (ii,jj)=oligo_pairs_ix[i,:]
            dG_mat[ii,jj] = dG_list[i]
            bps_mat[ii,jj] = bps_list[i]
            if oligo_list_2 is None: 
                dG_mat[jj,ii] = dG_list[i]
                bps_mat[jj,ii] = bps_list[i]
        return(bps_mat,dG_mat)
    else: 
        return(bps_list,dG_list)



def binding_dist_oilgo_lists(oligo_list_1,oligo_list_2 = '',hyb_conditions = None): 
    """
    calculate binding properties for set of oligos given nupacl hyb condition model. 

    inputs: 
    -------
    oligo_list_1 : list of oligos
    oligo_list_2 : option #1 - list of oligos to compare to 1
                   option #2 - 'self' (analyze self binding - a single oligo at a time)
                   option #3 - 'complement' oligo_list_1 is compared to it's complement
                   option #4 - 'self-hetero' oligo from list 1 are comapred to themselves (2 oligos)
    hyb_conditions : nupack model

    output: 
    -------
    bps - number of nt that are bound to something (not free)
    dG - reaction dG
    """
    # deal with special inputs values for oligo_list_2 (options #2 and #3 above)
    single_strand_flag = False
    if isinstance(oligo_list_2, str): 
        if oligo_list_2 == 'self-hetero': 
            oligo_list_2 = oligo_list_1.copy()
        elif oligo_list_2 == 'self': 
            single_strand_flag = True
        elif oligo_list_2 == 'complement':
            oligo_list_2 = [get_rcseq(o) for o in oligo_list_1]

    # validate inputs
    if not single_strand_flag and len(oligo_list_1) != len(oligo_list_2):
        raise ValueError("Oligo lists must be the same length")
    dGs = np.zeros(len(oligo_list_1))
    bps = np.zeros(len(oligo_list_1))
    for i in range(len(oligo_list_1)):
        if single_strand_flag: 
            strnds = [oligo_list_1[i]]
        else:
            strnds = [oligo_list_1[i],oligo_list_2[i]] 
        mfe_structures = nupack.mfe(strands=strnds, model=hyb_conditions)
        if len(mfe_structures)==0:
            raise ValueError('something is wrong - please check')
        dGs[i] = mfe_structures[0].energy
        structure_mat = mfe_structures[0].structure.matrix()
        if single_strand_flag: 
            bps[i] = (len(oligo_list_1[i]) - np.trace(structure_mat))/2
        else:
            # remove all self bindings
            structure_mat[:len(strnds[0]),:len(strnds[0])]=0
            structure_mat[len(strnds[1]):,len(strnds[1]):]=0 
            bps[i] = structure_mat.sum()/2
        
    return (bps,dGs)


def all_pairwise_pairing(oligos,hyb_conditions,avoid_self = False,return_matrices = False):
    """
    pairwise distances between oligos using nupack's MFE
    returns both dG and number of binding sites (bps)
    """
    # calculate mfe (bps and dG) for all oligos in the list including the pairing of self with self
    if avoid_self: 
        oligo_pairs = list(it.combinations(range(len(oligos)), 2))
    else: 
        oligo_pairs = [(i, j) for i, j in it.product(range(len(oligos)), repeat=2) if i <= j]
    
    bps = np.zeros(len(oligo_pairs))
    dGs = np.zeros(len(oligo_pairs))
    dG_mat = np.zeros((len(oligos),len(oligos)))
    bps_mat = np.zeros((len(oligos),len(oligos)))
    for i,(o1,o2) in enumerate(oligo_pairs):
        mfe_structures = nupack.mfe(strands=[oligos[o1],oligos[o2]], model=hyb_conditions)
        structure_mat = mfe_structures[0].structure.matrix()
        bps[i] = (len(oligos[o1]) + len(oligos[o2]) - np.trace(structure_mat))/2
        dGs[i] = mfe_structures[0].energy
        dG_mat[o1,o2]=dGs[i]
        dG_mat[o2,o1]=dGs[i]
        bps_mat[o1,o2]=bps[i]
        bps_mat[o2,o1]=bps[i]

    if return_matrices: 
        return(bps_mat,dG_mat)
    else:         
        return (oligo_pairs,bps,dGs)


def get_ATCG_freq(seq_list,atcg = ['A','T','C','G']):
    """
    performs hist count for each seq in list
    """
    L = max([len(s) for s in seq_list])
    Freqs = np.zeros((L,4))
    for i,n in enumerate(atcg):
        cnt = [s.count(n) for s in seq_list]
        Freqs[:,i]=np.bincount(cnt,minlength=L)
    return(Freqs)

def rand_seq(N,L,AT = {'min' : 0, 'max' :  float("inf")},
                min_counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0},
                max_counts = {'A': float("inf"), 'T': float("inf"), 'C': float("inf"), 'G': float("inf")},
                exclude_patterns = ['GGG', 'CCC', 'AAAA', 'TTTT'], 
                Tm = {'min' : 0, 'max' :  float("inf")},
                overhangs = ['',''],
                avoid = {'seq' : None, 'min_distance' : None},
                max_attempts = 1e6):
    """
    Goal: 
    The function generates random sequecnes based on rules. 

    Algo: 
    Uses rejection sampling, i.e. will keep trying until either it gets N seq or max_attempts were made. 

    Parameters:
    N (int): The number of random sequences to generate.
    L (int): The length of each random sequence.
    AT (dict): A dictionary with 'min' and 'max' keys to specify the minimum and maximum number of A/T nucleotides.
    min_counts (dict): A dictionary with keys 'A', 'T', 'C', and 'G' to specify the minimum counts of each nucleotide.
    max_counts (dict): A dictionary with keys 'A', 'T', 'C', and 'G' to specify the maximum counts of each nucleotide.
    exclude_patterns (list): A list of string patterns that should not appear in the generated sequences.
    Tm (dict): A dictionary with 'min' and 'max' keys to specify the minimum and maximum melting temperature (in Celsius) of the generated sequences.
    overhangs (list): A list of two strings to be added as prefix and suffix to the generated sequences.
    avoid (dict): A dictionary with 'seq' key to specify a list of sequences to avoid, and 'min_distance' key to specify the minimum Levenshtein distance to these sequences.
    max_attempts (int): The maximum number of attempts to generate a sequence before stopping.

    Returns:
    np.array: An array of generated sequences that satisfy the given constraints.

    """
    seqs = []
    if len(exclude_patterns) > 0:
        exclude_regex = re.compile('|'.join(exclude_patterns))
    else: 
        exclude_regex = re.compile(exclude_patterns[0])

    # update AT max based on requested length
    AT['max'] = min(AT['max'],L)
    # perform "rejection sampling" until we find enough sequences
    cnt=0
    Lix = list(range(L))
    while len(seqs)<N and cnt < max_attempts:
        cnt += 1
        candidate_seq = np.array(['N'] * L)
        AT_ix = np.random.choice(Lix,size = np.random.randint(AT['min'],AT['max']),replace = False)
        # AT_ix = random.sample(range(L),random.randint(AT['min'],AT['max']))
        candidate_seq[AT_ix] =  random.choices(['A','T'],k=len(AT_ix))
        GC_ix = np.flatnonzero(candidate_seq == 'N')
        candidate_seq[GC_ix] = random.choices(['G','C'],k=len(GC_ix))
        candidate_seq = ''.join(candidate_seq)
        candidate_seq = overhangs[0] + candidate_seq + overhangs[1]
        reject = False
        for letter in ['A','T','C','G']:
            letter_cnt = candidate_seq.count(letter) 
            if letter_cnt < min_counts[letter]:
                reject = True
            if letter_cnt > max_counts[letter]: 
                reject = True
        if exclude_regex.search(candidate_seq):
            reject = True
        if not reject:
            if Tm['min'] > 0 or Tm['max'] < 1000:
                tm = get_tm(candidate_seq,Na=20,dnac1=500)
            else: 
                tm=60
            if tm > Tm['min'] and tm <= Tm['max']:
                if avoid['seq'] is not None:
                    seq_to_avoid = avoid['seq'] 
                    min_distance = avoid['min_distance']
                    dist_to_seq_to_avoid = [len(s)-L+min_distance for s in seq_to_avoid]
                    dist_to_seq_to_avoid = np.array(dist_to_seq_to_avoid)
                    dist_to_seq_to_avoid = dist_to_seq_to_avoid[:,np.newaxis]
                    cand_to_avoid_dsts = lev_mat2(seq_to_avoid,candidate_seq)
                    if (cand_to_avoid_dsts>dist_to_seq_to_avoid).all():
                        seqs.append(candidate_seq)
                else: 
                   seqs.append(candidate_seq)

    if cnt == max_attempts: 
        print(f"only found {len(seqs)} seqeucnes in {cnt} attempts, please check requested conditions")           
    return(np.array(seqs))

def mutate_seqs(seqs,masks = None):
    """
    mutates sequences - good for random screening of variants for optimizations. 
    """
    if masks is None: 
        masks = [np.ones(len(seq),dtype=bool) for seq in seqs]
    new_seqs = []
    for i in range(len(seqs)):
        seq = seqs[i]
        ix_mut = random.choice(np.flatnonzero(masks[i]))
        seq = seq[:ix_mut] + random.choice(['A','C','T','G']) + seq[ix_mut+1:]
        new_seqs.append(seq)
    return(new_seqs)


def screen_oligos_with_levenshtein(current_oligos, candidate_oligos,min_dist):
    """
    screens a list of candidates oligos against a list of current oligos to make sure that they have min_dist from all current_oligos. 
    function makes sure to "update" current_oligo with any canidate that is legit, i.e. the to_keep set is not only min dist from all current_oligos
    it is also min_dist from all other to_keep entries
    """
    current_oligos = current_oligos.copy()
    to_keep = np.ones(len(candidate_oligos),dtype = bool)
    for i,cand in enumerate(candidate_oligos):
        for curr in current_oligos:
            cand_to_curr_dist = Levenshtein.distance(cand, curr)
            if cand_to_curr_dist < min_dist:
                to_keep[i] = False
                break
        if to_keep[i]:
            current_oligos.append(cand)
    return to_keep

def screen_oligos_with_nupack(current_oligos, candidate_oligos,nupack_model,min_dG = -3,max_bp = 5):
    """
    screens a list of candidates oligos against a list of current oligos to make sure that they have sufficient dG / bp bindings from all current_oligos. 
    function makes sure to "update" current_oligo with any canidate that is legit, i.e. the to_keep set is not only different from all current_oligos
    it is also different from all other to_keep entries according to the min_dG and max_bp criteria
    """
    current_oligos = current_oligos.copy()
    to_keep = np.ones(len(candidate_oligos),dtype = bool)
    for i,cand in enumerate(candidate_oligos):
        for j,curr in enumerate(current_oligos):
            bp,dG = binding_dist_oilgo_lists([curr],[cand],nupack_model)
            # _,bp,dG = all_pairwise_pairing([curr,cand],nupack_model,avoid_self=True)
            if dG[0] < min_dG or bp[0] > max_bp:
                to_keep[i] = False
                break
        if to_keep[i]:
            current_oligos.append(cand)
    return to_keep

def screen_oligos_based_on_attributes(candidate_oligos,attribute_dict):
    """
    Goal: check attributes of list of sequences. Attributes are checked based on dict of rules (attribute_dict)
    function will screen oligos to make sure that they have desired attributes. 
    
    Current supported attributes (keys in attrubute_dict) are: 
    GC content : tuple(list) of 2 values (GCmin,GCmax)
    patterns to exclude : list of regex patterns to exclude
    self_dG : tuple(list) of (dGmin,bp_max) of nupack secondary structure (MFE) dG is lowest cutoff and bp_max is higest
    Tm : tuple(list) of 2 values (Tm_max,Tm_max) of melting temeprature calcualted using get_tm with Na=20 and dnac1 = 500
    """
    to_keep = np.ones(len(candidate_oligos),dtype=bool)
    for attr_type,attr_rule in attribute_dict.items(): 
        if attr_type == "GC content":
            GCmin = attr_rule[0] 
            GCmax = attr_rule[1]
            freq = np.vstack([(s.count('A'),s.count('T'),s.count('G'),s.count('C')) for s in candidate_oligos]) 
            gc_cont = np.sum(freq[:,2:],axis=1)/np.sum(freq,axis=1)
            to_keep[gc_cont < GCmin] = False
            to_keep[gc_cont > GCmax] = False
        elif attr_type == "patterns to exclude":
            exclude_regex = re.compile('|'.join(attr_rule))
            pattern_found = [bool(exclude_regex.search(s)) for s in candidate_oligos]
            to_keep[pattern_found] = False
        elif attr_type == "self_dG": 
            min_dG_self = attr_rule[0][0]
            max_bps_self = attr_rule[0][1]
            nupack_model = attr_rule[1]
            for k,cand in enumerate(candidate_oligos):
                mfe_structures = nupack.mfe(strands=[nupack.Strand(cand,name='s')], model=nupack_model)
                structure_mat = mfe_structures[0].structure.matrix()
                self_bps = (len(cand) - np.trace(structure_mat))/2
                if mfe_structures[0].energy<min_dG_self or self_bps > max_bps_self:
                    to_keep[k]=False
        elif attr_type == "Tm":
            min_Tm = attr_rule[0]
            max_Tm = attr_rule[1]
            for k,cand in enumerate(candidate_oligos):
                tm = get_tm(cand,Na=20,dnac1=500) 
                if tm<min_Tm:
                    to_keep[k]=False
                if tm>max_Tm:
                    to_keep[k]=False
        else: 
            raise ValueError(f"oligo attribute: {attr_type} not supported")
    return to_keep


def find_ortho_oligo_set(L,N,nupack_model,oligos = None,seq_to_avoid = None,min_dist = 12,min_dG = -10,min_dG_self = -2, max_bp = 8, max_self_bp = 4,GCcont = (0.4,0.6),exclude_patterns = ['GGG', 'CCC', 'AAAA', 'TTTT'],verbose = True):
    """
    Goal: find a set of orthogonal oligos based on given rules. 
    screening is done using three set of criteria (attributes, lev, and nupack) so that it only checks things that are hard to compute if needed. 
    """
    if oligos is None: 
        oligos = list()

    start = time.time()
    if seq_to_avoid is None:
        oligos_start = 0
    else: 
        oligos = seq_to_avoid + oligos
        oligos_start = len(seq_to_avoid)

    nuc = np.array(['A','T','C','G'])
    def jn(s): 
        return(''.join(s))       
    for iter in range(N[1]):
        x = np.random.randint(4, size=(N[0], L))
        candidates= np.apply_along_axis(jn,axis=1,arr=nuc[x])
        to_keep = screen_oligos_based_on_attributes(candidates,{"GC content" : GCcont,"patterns to exclude" :exclude_patterns})
        candidates= candidates[to_keep]
        pass_attributes = np.sum(to_keep)

        if pass_attributes==0: 
            raise ValueError("did not find any oligo with reqruied attributes")

        # now we need to figure out if there is anything to screen against, 
        # if not (oligos is empty) find the first oligo that passes 2nd structure and continue
        if len(oligos)==0:
            to_keep = False
            cnt=-1
            while not to_keep:
                cnt+=1
                to_keep = screen_oligos_based_on_attributes(candidates[cnt],{"self_dG" : ((min_dG_self,max_self_bp),nupack_model)})
                to_keep = to_keep[0]

            oligos.append(candidates[cnt])
            continue
        # if oligo is not empty, continbue with edit dist, 2nd structure, and nupack binding
        else: 
            # screen using edit-distance
            to_keep = screen_oligos_with_levenshtein(oligos,candidates,min_dist = min_dist)
            candidates = candidates[to_keep]
            pass_leven = np.sum(to_keep)

            # screen based on oligo 2nd structure (avoid)
            to_keep = screen_oligos_based_on_attributes(candidates,{"self_dG" : ((min_dG_self,max_self_bp),nupack_model)})
            candidates = candidates[to_keep]
            pass_second_structure = np.sum(to_keep)

            # screen for nupack binding prediction 
            to_keep = screen_oligos_with_nupack(oligos,candidates,nupack_model,min_dG=min_dG,max_bp = max_bp)
            candidates = candidates[to_keep]
            pass_nupack = np.sum(to_keep)
            for cand in candidates:
                oligos.append(cand)
        if verbose: 
            print(f"{iter+1}/{N[1]}: From: {N[0]} " + 
                f"pass-attributes: {pass_attributes} "+
                f" pass leven: {pass_leven} "+
                f"pass 2nd strcutre: {pass_second_structure} "+
                f"pass nupack: {pass_nupack} "+
                f"total: {len(oligos)-oligos_start} time: {time.time()-start:.2f}")
                
    # remove any sequences that were added to oligos to make sure we avoid them... 
    oligos = oligos[oligos_start:]
    return oligos

  
def plot_strand_mfe_structure(strnds,nupack_model = None,ax = None,return_dG = False):
    """
    binary MFE matrix of multiple oligos with lines showing where the oligos are. 
    """
    s_l = np.cumsum([len(s) for s in strnds])
    mfe_structures = nupack.mfe(strands=strnds, model=nupack_model)
    structure_mat = mfe_structures[0].structure.matrix()
    if ax is None:
        plt.figure(figsize=(8,8))
        ax = plt.gca()
    sns.heatmap(structure_mat,ax=ax,cbar=False)
    for ln in s_l:
        ax.axhline(y=ln, color='r', linestyle='--',linewidth = 0.5)
        ax.axvline(x=ln, color='r', linestyle='--',linewidth = 0.5)
    ax.set_xticks(s_l)
    ax.set_xticklabels([str(t) for t in s_l])
    ax.set_yticks(s_l)
    ax.set_yticklabels([str(t) for t in s_l])
    ax.set_title(f"dG={mfe_structures[0].energy:.2f}")
    if return_dG:
        return(mfe_structures[0].energy)


def to_html_color(seq, pattern_dict, show_reverse_complement=False):
    """
    outputs seq to html so in could show up colored in a notebook. 
    pattern_dict has the seq (keys) and what colors they should have (values)
    """
    def colorize_substring(substring, color):
        return f'<span style="color: {color};">{substring}</span>'

    def colorize_sequence(sequence, pattern_dict):
        colored_seq = sequence
        for pattern, color in pattern_dict.items():
            start = 0
            while start < len(colored_seq):
                index = colored_seq.find(pattern, start)
                if index != -1:
                    colored_substring = colorize_substring(pattern, color)
                    colored_seq = colored_seq[:index] + colored_substring + colored_seq[index+len(pattern):]
                    start = index + len(colored_substring)
                else:
                    break
        return colored_seq

    top_strand = seq
    bottom_strand = get_rcseq(seq) if show_reverse_complement else None

    top_strand_colored = colorize_sequence(top_strand, pattern_dict)
    bottom_strand_colored = colorize_sequence(bottom_strand, pattern_dict) if bottom_strand else None

    html_output = f'<pre>{top_strand_colored}\n'

    if bottom_strand_colored:
        html_output += f'{bottom_strand_colored}'

    html_output += '</pre>'

    return html_output
