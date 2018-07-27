
# coding: utf-8

# In[44]:

#from pyplink import PyPlink
import numpy as np
import pandas as pd
import itertools
import scipy.spatial.distance
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import glob
import random
from random import shuffle
import os.path


# In[45]:

import msprime
import allel


# In[46]:

import simuPOP
import simuPOP.utils


# In[47]:

from pylab import rcParams
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# In[48]:

def make_rec_map(n_chrom, len_chrom, recombination_rate):
    n_chrom = int(n_chrom)
    len_chrom = int(len_chrom)
    A = [[x, x+1] for x in range(int(len_chrom), int(len_chrom)*(n_chrom+1), int(len_chrom))]
    rm_pos = [0] + list(itertools.chain.from_iterable(A))
    B = [recombination_rate, 0.5] * n_chrom
    rec_rates = B + [0]
    rec_map = msprime.RecombinationMap(rm_pos, rec_rates)
    return(rec_map)


# In[ ]:




# In[67]:

def sim_chrom(
    n_chrom,
    chrom_len,
    recombination_rate,
    mutation_rate,
    sample_size,
    Ne_ancestral,
    Ne_recent,
    Ne_switchdate,
    print_history = False):

    #recombination map
    rec_map = make_rec_map(n_chrom, chrom_len, recombination_rate)
    
    #population_configurations = [
    #    msprime.PopulationConfiguration(
    #        sample_size=sample_size, initial_size=Ne_recent)
    #]
    demographic_events = [
        msprime.PopulationParametersChange(
            time=Ne_switchdate, initial_size=Ne_ancestral, population_id=0)
    ]
    # Use the demography debugger to print out the demographic history
    # that we have just described.
    #dp = msprime.DemographyDebugger(
    #    Ne=Ne_recent,
    #    population_configurations=population_configurations,
    #    demographic_events=demographic_events)
    #if print_history:
    #    dp.print_history()
    #   return None
    #else:
    tree_sequence = msprime.simulate(
                sample_size = sample_size, 
                Ne = Ne_recent,
                recombination_map = rec_map,
                mutation_rate=mutation_rate,
                #population_configurations = population_configurations,
                demographic_events = demographic_events,
                #random_seed = random_seed
                            )
    shape = tree_sequence.get_num_mutations(), tree_sequence.get_sample_size()
    print 'num_mutations: {}, samples: {}'.format(*shape)
    return(tree_sequence)


# In[ ]:




# n_chrom = 6
# chrom_len = 1e5
# mutation_rate = 1e-8
# recombination_rate = 1e-8
# sample_size = 20
# Ne_recent = 1000
# Ne_ancestral = 1000
# Ne_switchdate = 100
# 
# rec_map = make_rec_map(n_chrom, chrom_len, recombination_rate)
#     
# population_configurations = [
#         msprime.PopulationConfiguration(
#             sample_size=sample_size, initial_size=Ne_recent)]
# 
# demographic_events = [
#         msprime.PopulationParametersChange(
#             time=Ne_switchdate, initial_size=Ne_ancestral, population_id=0)]

# msprime.simulate(
#                     recombination_map = rec_map,
#                     mutation_rate=mutation_rate,
#                     population_configurations = population_configurations,
#                     demographic_events = demographic_events,
#                                 )
# 

# # Filter for allele freqeuncy

# In[ ]:

def filter_allele_freq(sim_data, MAF):
    assert(MAF <= .5)
    low = MAF
    high = 1.0 - MAF
    keep = []
    for tree in sim_data.trees():
            for site in tree.sites():
                mutation = site.mutations[0] # assume just one
                p = tree.get_num_leaves(mutation.node) / np.float(tree.get_sample_size()) # frequency
                if p >= low and p <= high:
                    keep.append(site)
    print len(keep)
    return(sim_data.copy(keep))


# In[68]:

def downsample_sim_data(sim_data, max_loci):
    sites = []
    for tree in sim_data.trees():
            for site in tree.sites():
                #mutation = site.mutations[0] # assume just one
                sites.append(site)
    
    if len(sites) > max_loci:
        picks = sorted(np.random.choice(len(sites), replace = False, size = max_loci))
        keep = [sites[i] for i in picks]
        print len(keep)
        return(sim_data.copy(keep))
    else:
        return(sim_data)


# In[ ]:

# get the LD in the coalescent data


# In[ ]:

def get_msprime_ld(sim_data):
    ld_calc = msprime.LdCalculator(sim_data)
    A = ld_calc.get_r2_matrix()
    plt.imshow(A, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    plt.xticks([])
    plt.yticks([])
    plt.show()


# In[ ]:

def get_genotypes(tree_sequence):
    shape = tree_sequence.get_num_mutations(), tree_sequence.get_sample_size()
    A = np.empty(shape, dtype="u1")
    count = 0
    for variant in tree_sequence.variants():
        A[variant.index] = variant.genotypes
        count+=1
    return(A)


# In[ ]:

def get_diploid_genotypes(tree_sequence, pop_index = 0):
    target_samples = tree_sequence.get_samples(pop_index)
    assert len(target_samples) % 2 == 0
    geno_array = get_genotypes(tree_sequence)
    target_genotypes = geno_array[:,target_samples]
    return (target_genotypes + np.roll(target_genotypes,-1, axis=1))[:,::2]


# In[ ]:

def get_genotype_ld(sim_data):
    first_r2_mat = allel.stats.ld.rogers_huff_r_between(get_diploid_genotypes(sim_data), get_diploid_genotypes(sim_data), fill=np.nan)
    plt.imshow(first_r2_mat, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    plt.xticks([])
    plt.yticks([])
    plt.show()


# In[ ]:

def get_chromosome_positions(sim_data, chromsome_length):
    positions = np.array([pos for pos,node,idx in sim_data.mutations()])
    chromosome_positions = np.mod(positions, chromsome_length) # modulo with the chromosome length
    return (chromosome_positions)


# # get haplotypes is a mess, could be faster

# In[ ]:

def get_haplotypes(sim_data):
    haplotypes = list(sim_data.haplotypes())
    haplotypes = [list(xx) for  xx in haplotypes]
    haplotypes = np.array(haplotypes).astype(np.int) 
    haplotypes = [list(xx) for  xx in haplotypes]
    return(haplotypes)


# In[ ]:

def get_loci_per_chromosome(chromosome_positions):
    breaks = list(np.where(np.diff(chromosome_positions)<0)[0])
    breaks = [-1] + breaks + [len(chromosome_positions)-1]
    loci_per_chromosome = [xx for xx in np.diff(breaks)]
    return(loci_per_chromosome)


# # simuPOP

# # controlled  mating

# In[ ]:

def parent_shuffle(par):
    males = [x[0] for x in par]
    females = [x[1] for x in par]
    shuffle(males)
    shuffle(females)
    return(zip(males, females))

def fixedChooser(pop):
    popSize = pop.popSize()
    #print('Entered fixedChooser')
    #controlled_matting_log.info('Entered fixedChooser')
    # identify males and females    
    male_dict = dict()
    female_dict = dict()
    for indx, ind in enumerate(pop.individuals()):
        if ind.sex() == simuPOP.MALE:
            male_dict[indx] = ind
        elif ind.sex() == simuPOP.FEMALE:
            female_dict[indx] = ind  
    #print("Populated dictionaries")
    #print('males_dict', male_dict.keys())
    #print('females_dict', female_dict.keys())  
    #print("parent males: {}, parent females: {}".format(len(male_dict), len(female_dict) ))
    file_parents = list()
    pick_filename = "/home/ryan/simNe/parent_pick/{}_{}.pick".format(int(popSize), int(popSize))
    #print(pick_filename)
    #print(os.path.isfile(pick_filename))
    # file represents the 0-popsize/2 males/females in each dict
    with open(pick_filename) as INFILE:
        for cnt, line in enumerate(INFILE):
            parents_from_file = [int(xx) for xx in line.strip().split()]
            #print (cnt)
            file_parents.append(parents_from_file)

    # randomize these parent pairs    
    file_parents = parent_shuffle(file_parents)
    
    for cnt, file_parent_pair in enumerate(file_parents):
        #print ('offspring: {}, file male: {}, file female: {}'.format(cnt, file_parent_pair[0], file_parent_pair[1] ))
        pop_male = male_dict.keys()[file_parent_pair[0]]
        pop_female = female_dict.keys()[file_parent_pair[1]]
        #print('offspring: {}, pop male: {}, pop female: {}'.format(cnt, pop_male, pop_female))
        if pop_male in male_dict and pop_female in female_dict: 
            #print ('Found!')
            yield pop_male, pop_female
        else:
            #print('Not Found')
            yield(None)


# # make the inital population

# In[ ]:

def get_export_genotypes(pop, n_ind):
    """get a numpy array of genotypes to export.  
    n_ind is the number of indiviudals to select at random"""
    n_loci = len(pop.lociNames())
    # will hold genotypes
    genos = np.zeros((n_ind, n_loci))
    picked = np.random.choice(a = pop.popSize(), size = n_ind, replace = False)
    genos_idx = 0
    for idx, ind in enumerate(pop.individuals()):
        if idx in picked:
            geno_c1 = np.array(ind.genotype(ploidy = 0))
            geno_c2 = np.array(ind.genotype(ploidy = 1))
            geno = geno_c1+geno_c2
            genos[genos_idx] = geno
            genos_idx +=1
    return(genos)


# In[ ]:

def do_forward_sims(
        sim_data,
        chrom_len, 
        diploid_Ne, 
        batchname,
        repilcates, 
        simupop_seed):
    print("start getting chromosomes positions")
    chromosome_positions = get_chromosome_positions(sim_data = sim_data, chromsome_length = chrom_len)
    print("done getting chromosomes positions")

    haplotypes = get_haplotypes(sim_data)
    loci_per_chromsome = get_loci_per_chromosome(chromosome_positions)
    n_chrom = len(loci_per_chromsome)
    # set up the ancestral pop in simuPOP
    initial = simuPOP.Population(diploid_Ne, # here is the diploid number
                            loci= loci_per_chromsome,  # should be the number of loci on each chromosome
                            lociPos=list(chromosome_positions),
                            ploidy = 2,
                            infoFields=['father_idx', 'mother_idx', 'ind_id'],
                            #alleleNames=['A', 'C', 'G', 'T'],
                            lociNames = ['locus_{}'.format(x) for x in xrange(len(chromosome_positions))])
    simuPOP.initGenotype(initial, prop=[1.0/len(haplotypes)]*len(haplotypes), haplotypes=list(haplotypes))
    simuPOP.tagID(initial, reset = 1)
    initial_export = get_export_genotypes(initial, initial.popSize())
    np.savetxt('./share/{}/Ne-{}_Chr-{}/Ne-{}_Chr-{}.inital.txt'.format(batchname, diploid_Ne, n_chrom, diploid_Ne, n_chrom), initial_export,  delimiter = '\t', fmt = '%01d')
    
    # map-ped
    simuPOP.utils.export(initial, format='PED',output='./share/{}/Ne-{}_Chr-{}/Ne-{}_Chr-{}.inital.ped'.format(batchname, diploid_Ne, n_chrom, diploid_Ne, n_chrom), 
                         gui=False, idField = 'ind_id')
    simuPOP.utils.export(initial, format='MAP',output='./share/{}/Ne-{}_Chr-{}/Ne-{}_Chr-{}.inital.map'.format(batchname, diploid_Ne, n_chrom, diploid_Ne, n_chrom), 
                         gui=False)
    
    # Doesn't yet work!!
    # set the seed for simuPOP
    #simuPOP.setRNG(seed=simupop_seed)
    # and in Python
    #random.seed(simupop_seed)
    # and for numpy
    #np.random.seed(simupop_seed)
    
    print ("initalizing the forward simulator")
    simu = simuPOP.Simulator(simuPOP.Population(diploid_Ne, # here is the diploid number
                            loci=get_loci_per_chromosome(chromosome_positions),  # should be the number of loci on each chromosome
                            lociPos=list(chromosome_positions),
                            ploidy = 2,
                            infoFields=['ind_id', 'father_idx', 'mother_idx'],
                            #alleleNames=['A', 'C', 'G', 'T'],
                            lociNames = ['locus_{}'.format(x) for x in xrange(len(chromosome_positions))]
                        ), 
                         rep=repilcates
                        )
    
    print ("Start evolving {} replicates".format(repilcates))    
    simu.evolve(
        initOps=[
            simuPOP.InitSex(sex = [simuPOP.MALE, simuPOP.FEMALE]), # alternate sex
            simuPOP.InitGenotype(prop=[1.0/len(haplotypes)]*len(haplotypes), haplotypes=list(haplotypes))
        ],
        matingScheme=simuPOP.HomoMating(
            chooser = simuPOP.PyParentsChooser(fixedChooser),
            generator = simuPOP.OffspringGenerator(
                sexMode=(simuPOP.GLOBAL_SEQUENCE_OF_SEX, simuPOP.MALE, simuPOP.FEMALE),
            ops = [simuPOP.Recombinator(intensity = 1.0/chrom_len), 
                   simuPOP.ParentsTagger()
            ]),
        ),
        postOps=[],
        gen = 20
    )
    
    print ("Done evolving {} replicates!".format(repilcates))
    
    
    # export the data
    print ("Exporting data!".format(repilcates))
    for rep, pop in enumerate(simu.populations()):
        if diploid_Ne >= 200:
            pop_genotypes = get_export_genotypes(pop, n_ind = 200) #  select 200 inds
        else:
            pop_genotypes = get_export_genotypes(pop, n_ind = diploid_Ne) #  select 200 inds

        np.savetxt('./share/{}/Ne-{}_Chr-{}/Ne-{}_Chr-{}_Frep-{}.geno'.format(batchname, diploid_Ne, n_chrom, diploid_Ne, n_chrom, rep).format(rep), 
                   pop_genotypes ,  delimiter = '\t', fmt = '%01d')
        if rep%10 == 0:
            print "saved rep {}".format(rep)
        
        
#        # subset the data
#        current_pop = pop.clone()
#        subset_pop = current_pop.extractIndividuals(indexes=sorted(list(np.random.choice(a = diploid_Ne, size = 200, replace = False)))) # make a 'new' pop from a subset of the individuals
#        simuPOP.utils.export(subset_pop, format='csv', output='./share/{}/Ne-{}_Chr-{}/Ne-{}_Chr-{}_Frep-{}.geno'.format(batchname, diploid_Ne, n_chrom, diploid_Ne, n_chrom, rep), 
#                        gui=False, header  = False, delimiter = '\t', 
#                        affectionFormatter = None, sexFormatter = None,
#                        genoFormatter = {(0,0):0, (0,1):1, (1,0):1, (1,1):2})

    

