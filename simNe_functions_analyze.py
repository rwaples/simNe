
# coding: utf-8

# # Post - Sim Analysis 

# In[7]:

#from pyplink import PyPlink
import numpy as np
import pandas as pd
import scipy as sp
import itertools
import scipy.spatial.distance
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import glob
import random
from random import shuffle
import os.path


# In[8]:

import msprime
import allel


# In[9]:

import simuPOP

import simuPOP.utils


# In[10]:

from pylab import rcParams
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# # Functions to 

# In[6]:

def get_rsq(geno_mat):
    """returns the squared pearson r for each pair of loci in a condensed distance matrix"""
    return scipy.spatial.distance.pdist(geno_mat.T, lambda x, y: scipy.stats.pearsonr(x, y)[0])**2


# In[ ]:

def get_r2_fast(geno_mat):
    norm_snps = (geno_mat - sp.mean(geno_mat, 0)) / sp.std(geno_mat, 0, ddof=1)
    norm_snps = norm_snps.T
    num_snps, num_indivs = norm_snps.shape
    ld_mat = (sp.dot(norm_snps, norm_snps.T) / float(num_indivs-1))**2
    return(ld_mat)

def get_overall_mean(ld_mat):
    return(ld_mat[np.triu_indices_from(ld_mat, k=1)].mean())

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return itertools.izip(a, a)

def get_nonoverlapping_mean(ld_mat):
    # randomly selects non-overlapping pairs of loci and gets their mean
    n_loci = ld_mat.shape[1]
    indexes = range(n_loci)
    random.shuffle(indexes)
    a = indexes[0 : n_loci/2]
    b = indexes[n_loci/2 : n_loci]
    dat = ld_mat[a,b]
    return (dat.mean())

def shuffle_mat(mat):
    # shuffles a square matrix by rows and columns, while keeping the association
    shape1, shape2 = mat.shape 
    assert(shape1 == shape2)
    new_order = range(shape1)
    random.shuffle(new_order)
    return(mat[new_order][:,new_order])

def get_block_r2(ld_mat, nblocks):
    # shuffles the matrix, then divides it into nblocks and takes the average of the meanr r2 within each block 
    ld_mat = shuffle_mat(ld_mat)
    blocks = np.array_split(range(ld_mat.shape[1]), nblocks)
    means = []
    for block in blocks:
        submat = ld_mat[block][:,block]
        means.append(get_overall_mean(submat))
        
    sum_squares = np.sum(map(lambda x: x**2, means))
    #    print submat
    #print means 
    return(np.mean(means), sum_squares)


# In[ ]:

def get_weirFst(twopop_geno_mat, popsize):
    ga = to_Genotype_array(twopop_geno_mat)
    a, b, c = allel.weir_cockerham_fst(ga, subpops=[range(popsize), range(popsize, popsize*2)])
    fst = np.sum(a) / (np.sum(a) + np.sum(b) + np.sum(c))
    return fst


# In[ ]:

def to_Genotype_array(geno_mat):
    a = np.where(geno_mat>0, 1, 0)
    b = np.where(geno_mat>1, 1, 0)
    return (allel.GenotypeArray(np.stack([a,b], axis =2)))


# In[ ]:

#allel.GenotypeArray(np.stack([a,b], axis =2))


# In[ ]:

# function to calculate temporal F given an ancestral and target pop
def get_temporalF(ancestral, current):
    """
    Nei and Tajima 1981 temporal F 
    """
    #make sure the loci in each pop line up!
    P1 = ancestral.mean(0)/2.
    P2 = current.mean(0)/2.
    PP = (P1+P2)/2.0
    TF_num = (P2-P1)**2
    TF_denom = (PP - P1*P2)
    TF_unweighted = (TF_num/TF_denom).mean() # mean of the ratios
    TF_weighted = TF_num.sum()/TF_denom.sum() # ratio of the sums

    return(TF_unweighted, TF_weighted)


# In[ ]:

def get_Jorde_Ryman_F(ancestral, current):
    xi = ancestral.mean(0)/2.
    yi = current.mean(0)/2.
    zi = (xi+yi)/2.
    num = ((xi - yi)**2).sum()
    denom = (zi*(1.0-zi)).sum()
    return(num/denom)


# In[ ]:

def get_Nei_Gst(pop1, pop2):
    P1 = pop1.mean(0)/2.
    P2 = pop2.mean(0)/2.
    
    sP1 = 1.0 - (P1**2 + (1-P1)**2)
    sP2 = 1.0 - (P2**2 + (1-P2)**2)
    Hexp = (sP1+sP2)/2.0
    Pbar = (P1 + P2)/2.0
    Htot = 1 - Pbar**2 - (1-Pbar)**2
    F = 1.0 - Hexp/Htot
    Fst_u = F.mean() # unweighted
    
    G_num = Htot - Hexp
    G_denom = Htot
    Fst_w = G_num.sum()/G_denom.sum() # weighted
    #return(P1, P2)
    #return(F)
    #return(F, G)
    return(Fst_u, Fst_w)


# In[ ]:

#test_pop1 = np.loadtxt('./share/feb25a/Ne-400_Chr-1/Ne-400_Chr-1_Frep-0.geno', dtype = 'int8', delimiter='\t') 
#test_pop2 = np.loadtxt('./share/feb25a/Ne-400_Chr-1/Ne-400_Chr-1_Frep-1.geno', dtype = 'int8', delimiter='\t')
#test_ancestral_pop = np.loadtxt('./share/feb25a/Ne-400_Chr-1/Ne-400_Chr-1.inital.txt', dtype = 'int8', delimiter='\t')


# In[ ]:

def get_allele_counts(geno_array):
    """Used for calcualting Fst and the (2d) SFS
    rows are loci, 
    columns are counts of specific alleles, sum of each row = sample size
    """
    n_samples = geno_array.shape[0]
    #print n_samples
    derived_counts = geno_array.sum(axis = 0) # sum alleles over indiviudals 
    ancestral_counts = (n_samples*2) - derived_counts
    #return derived_counts
    return np.stack((ancestral_counts, derived_counts)).T


# In[ ]:

def get_Hudson_Fst(pop1, pop2):
    ac1 = get_allele_counts(pop1)
    ac2 = get_allele_counts(pop2)
    num, denom = allel.stats.hudson_fst(ac1, ac2)
    fst_overall = np.sum(num)/np.sum(denom)
    #print fst_overall
    #return(num, denom)
    return(fst_overall)


# In[ ]:

def get_Patterson_f2(pop1, pop2):
    """numerator from hudsons Fst"""
    ac1 = get_allele_counts(pop1)
    ac2 = get_allele_counts(pop2)
    return allel.stats.admixture.patterson_f2(ac1, ac2).mean()


# In[ ]:

def get_2dSFS(pop1, pop2):
    ac1 = get_allele_counts(pop1)
    ac2 = get_allele_counts(pop2)
    return(allel.stats.sf.joint_sfs_folded(ac1, ac2))


# In[ ]:

def do_analysis(target_dir,  L = [16, 64, 256], S = [25, 50, 100, 200], min_AC = 2, replicates = 1):
    """do_analysis should be pointed at a directory, 
    this will make it easier to manage the target replicate, ancestral pop, and pairs of populations"""
    
    results_file = os.path.join(target_dir, 'results.txt')
    with open(results_file, 'w') as RESULTS:
        #RESULTS.write('file\tL\tS\tSrep\tmean_r2\tLocus_pairs\n')
        RESULTS.write('\t' .join(['file','L','S','Srep', 'actual_pairs_r2', 'overall_r2', 'block_r2', 'block_sum_squares', 'nonoverlap_r2', 
                                  'temporalF_u', 'temporalF_w', 'Patterson_f2', 'Hudson_fst', 'Jorde_Ryman_F', 'weir_Fst', 'actual_nloci_temporal'
                                 ]))
        RESULTS.write('\n')

    ancestral_file = glob.glob(os.path.join(target_dir, '*.inital.txt'))
    assert(len(ancestral_file)==1)
    ancestral_file = ancestral_file[0]
    ancestral_data = np.loadtxt(ancestral_file, dtype = 'int8', delimiter='\t')
    ancestral_nind = ancestral_data.shape[0]
    
    Frep_file_list = glob.glob(os.path.join(target_dir, '*.geno'))
    print('found {} *.geno files'.format(len(Frep_file_list)))
    cnt = 0
    for infile in Frep_file_list:
        cnt += 1
        if cnt % 20 == 0:
            print ("working on file: {}".format(infile))
        base_data = np.loadtxt(infile, dtype = 'int8', delimiter='\t')
        rep_nind = base_data.shape[0]
        for n_loci in L:
            for n_sample in S:
                for Srep in range(replicates):
                    # pick individuals for the rep and ancestral 
                    # each rep file has up to 200 inds
                    if n_sample <= rep_nind:
                        pick_ind_rep = np.sort(np.random.choice(a = rep_nind, size = n_sample, replace = False))
                        pick_ind_anc = np.sort(np.random.choice(a = ancestral_nind, size = n_sample, replace = False))

                        subset_data = base_data[pick_ind_rep,:]
                        # do the ancestral data matching the sample size, and as a awhole
                        #ancestral_full = ancestral_data.copy()
                        ancestral_subset = ancestral_data[pick_ind_anc,:]

                        # filter for Allele count (AC) based on the sample
                        rep_AC = subset_data.sum(0)
                        both_AC = rep_AC + ancestral_subset.sum(0) # sum of AC in the acestral and base

                        max_rep_AC = 2 * n_sample - min_AC
                        max_both_AC = 4 * n_sample - min_AC

                        lower_rep_test = min_AC <= rep_AC
                        upper_rep_test = max_rep_AC >= rep_AC
                        lower_both_test = min_AC <= both_AC
                        upper_both_test = max_both_AC >= both_AC

                        passed_AC_rep = np.logical_and(lower_rep_test, upper_rep_test)
                        passed_AC_both = np.logical_and(lower_both_test, upper_both_test)

                        # Pick the set of loci passing MAC filters
                        # also subset the ancestral data in the same way!!
                        r2_input = subset_data[:, passed_AC_rep]

                        temporal_rep = subset_data[:, passed_AC_both]
                        temporal_anc = ancestral_subset[:, passed_AC_both]
                        #ancestral_full = ancestral_full[:,passed]

                        remaining_loci_r2 = r2_input.shape[1]
                        remaining_loci_temporal = temporal_rep.shape[1]

                        if remaining_loci_r2 > n_loci:
                            pick_loci_r2 = np.sort(np.random.choice(a = remaining_loci_r2, size = n_loci, replace = False))
                            r2_input = r2_input[:, pick_loci_r2]

                        actual_nloci_r2 = r2_input.shape[1]
                        actual_pairs_r2 = (actual_nloci_r2 * (actual_nloci_r2-1))/2
                        r2_mat = get_r2_fast(r2_input)

                        block_r2, block_sum_squares = get_block_r2(r2_mat, 16)
                        nonoverlap_r2 = get_nonoverlapping_mean(r2_mat)
                        overall_r2 = get_overall_mean(r2_mat)

                        if remaining_loci_temporal > n_loci:
                            pick_loci_temporal = np.sort(np.random.choice(a = remaining_loci_temporal, size = n_loci, replace = False))
                            temporal_rep = temporal_rep[:, pick_loci_temporal]
                            temporal_anc = temporal_anc[:, pick_loci_temporal]

                        actual_nloci_temporal = temporal_anc.shape[1]

                        # actually do the temporal calculations
                        temporalF_u, temporalF_w  = get_temporalF(temporal_anc, temporal_rep)
                        Pf2 = get_Patterson_f2(temporal_anc, temporal_rep)
                        Hfst = get_Hudson_Fst(temporal_anc, temporal_rep)
                        Jorde_Ryman_F = get_Jorde_Ryman_F(temporal_anc, temporal_rep)
                        forWeir = np.concatenate([temporal_anc, temporal_rep], axis =0).T
                        weir_Fst = get_weirFst(forWeir, popsize = n_sample)

                        #SFS = get_2dSFS(ancestral_full, subset_data)
                        #SFS = SFS.flatten()

                        with open(results_file, 'a') as RESULTS:
                            RESULTS.write('{}\t'.format(infile))
                            for xx in [n_loci, n_sample, Srep, actual_pairs_r2, overall_r2, block_r2, block_sum_squares, nonoverlap_r2]:
                                RESULTS.write('{}\t'.format(xx))
                            for xx in [temporalF_u, temporalF_w, Pf2, Hfst, Jorde_Ryman_F, weir_Fst]:
                                RESULTS.write('{}\t'.format(xx))
                            for xx in [actual_nloci_temporal]:
                                RESULTS.write('{}'.format(xx))
                            #for xx in [SFS]:
                            #    RESULTS.write(','.join([str(x) for x in SFS]))
                            #    RESULTS.write('\t')                            
                            RESULTS.write('\n')
            
    # pairwise analysis of populations
    pair_results_file = os.path.join(target_dir, 'results.pairs.txt')
    with open(pair_results_file, 'w') as PAIR_RESULTS:
        PAIR_RESULTS.write('\t' .join(['file1','file2', 'S', 'L', 'Srep',
                             'Patterson_f2', 'Hudson_fst', 'weir_Fst', 'NeiGst_unweghted', 'NeiGst_weghted', 'actual_loci'
                             ]))
        PAIR_RESULTS.write('\n')

    for file1, file2 in zip(Frep_file_list,Frep_file_list[1:])[::2]:
        pop1_data = np.loadtxt(file1, dtype = 'int8', delimiter='\t')
        pop2_data = np.loadtxt(file2, dtype = 'int8', delimiter='\t')
        pop1_nind = pop1_data.shape[0]
        pop2_nind = pop2_data.shape[0]
        #print file1, file2
        for n_loci in L:
            for n_sample in S:
                for Srep in range(replicates):
                    if n_sample <= rep_nind:
                        pick_inds = np.sort(np.random.choice(a = pop1_nind, size = n_sample, replace = False))
                        subset_pop1 = pop1_data[pick_inds, :]
                        pick_inds = np.sort(np.random.choice(a = pop2_nind, size = n_sample, replace = False))
                        subset_pop2 = pop2_data[pick_inds, :]

                        # filter for combined MAC
                        both_AC = subset_pop1.sum(0) + subset_pop2.sum(0)
                        max_both_AC = 4 * n_sample - min_AC
                        lower_both_test = min_AC <= both_AC
                        upper_both_test = max_both_AC >= both_AC
                        passed_AC_both = np.logical_and(lower_both_test, upper_both_test)

                        subset_pop1 = subset_pop1[:, passed_AC_both]
                        subset_pop2 = subset_pop2[:, passed_AC_both]

                        # random subset of loci
                        remaining_loci = subset_pop1.shape[1]
                        if remaining_loci > n_loci:      
                            pick_loci = np.sort(np.random.choice(a = remaining_loci, size = n_loci, replace = False))
                            subset_pop1 = subset_pop1[:,pick_loci]
                            subset_pop2 = subset_pop2[:,pick_loci]
                        actual_loci = subset_pop1.shape[1]

                        # statistics
                        Pf2 = get_Patterson_f2(subset_pop1, subset_pop2)
                        Hfst = get_Hudson_Fst(subset_pop1, subset_pop2)
                        
                        forWeir = np.concatenate([subset_pop1, subset_pop2], axis =0).T
                        weir_Fst = get_weirFst(forWeir, popsize = n_sample)
                        
                        NeiGst_unweighted, NeiGst_weighted = get_Nei_Gst(subset_pop1, subset_pop2)


                        # TODO add Hudson Fst back in
                        with open(pair_results_file, 'a') as PAIR_RESULTS:
                            for xx in [file1, file2, n_sample, n_loci, Srep,
                                      Pf2, Hfst, weir_Fst, NeiGst_unweighted, NeiGst_weighted]:
                                PAIR_RESULTS.write('{}\t'.format(xx))
                            for xx in [actual_loci]:
                                PAIR_RESULTS.write('{}'.format(xx))
                            PAIR_RESULTS.write('\n')


# # Other

# In[ ]:

def get_genotype_ld(sim_data):
    first_r2_mat = allel.stats.ld.rogers_huff_r_between(get_diploid_genotypes(sim_data), get_diploid_genotypes(sim_data), fill=np.nan)
    plt.imshow(first_r2_mat, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    plt.xticks([])
    plt.yticks([])
    plt.show()


# In[ ]:

# Not used !!!
def wrap_r2(file_list, output_file, MAF, L = [16, 64, 256, 1024, 4096], S = [50, 100], replicates = 1):
    with open(output_file, 'w') as OUTFILE:
        OUTFILE.write('file\tL\tS\tSrep\tmean_r2\tLocus_pairs\n') # write header
    for infile in file_list:
        print "working on file: {}".format(infile)
        base_data = np.loadtxt(infile, dtype = 'int8', delimiter='\t')
        for n_loci in L:
            for n_sample in S:
                for Srep in range(replicates):
                    # sample inds
                    pick_inds = np.sort(np.random.choice(a = 200, size = n_sample, replace = False))
                    subset_data = base_data[pick_inds,:]
                    # filter for MAF based on the freqs in the sample
                    subset_data = filter_MAF(subset_data, MAF = MAF)
                    # select n_loci from among those loci passing MAF filters
                    remaining_loci = subset_data.shape[1]
                    pick_loci = np.sort(np.random.choice(a = remaining_loci, size = n_loci, replace = False))
                    subset_data = subset_data[:,pick_loci]
                    actual_loci = subset_data.shape[1]
                    n_locus_pairs = (actual_loci * (actual_loci-1))/2
                    #print pick_inds, pick_loci
                    #return(subset_data)
                    mean_rsq = get_r2_fast(subset_data).mean()
                    with open(output_file, 'a') as OUTFILE:
                        OUTFILE.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(infile, n_loci, n_sample, Srep, mean_rsq, n_locus_pairs))


# ## Robin's code
# GetFst <- function(yy)     {
# OldP = yy[1,]
# P2 = yy[2,]
# Fst = seq(1:length(L))
# SumP1 = 1 - (P2^2 + (1-P2)^2)
# SumP2 = 1 - (OldP^2 + (1-OldP)^2)
# Hexp = (SumP1+SumP2)/2
# Pbar = (P2 + OldP)/2
# Htot = 1 - Pbar^2 - (1-Pbar)^2 
# F = 1 - Hexp/Htot
# Fst[1] = mean(F,na.rm=TRUE)
#   for (k in 2:length(L))  {
#     FF = F[1:L[k]]
#     Fst[k] = mean(FF)  }  # end for k                            
# return(Fst)  } # end function
# 
# is.odd <- function(x) x %% 2 != 0
