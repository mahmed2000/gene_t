####################################################################################
#
#   Generates benign variant genes using ensembl's variant tables
#
#   Required:
#       Correct directory structure as in README
#       Packages: pandas
#       Param: Name of directory to generate for (eg, TP53 in example branch)
#
#
####################################################################################

import pandas, glob, random, sys, re, warnings, os


def gen_genomes():
    # get variants, and mutation locations, sorted by locations
    alleles = list(VARS.sort_values(by='Chromosome/scaffold position start (bp)')['Variant alleles'].str.split('/'))
    allele_index = list(VARS.sort_values(by='Chromosome/scaffold position start (bp)')['Chromosome/scaffold position start (bp)'])
    os.mkdir(f".{os.sep}{GENE}{os.sep}normal{os.sep}")
    for i in range(1, num_samples):
        full_seq = BASE_GENOME
        # for n random mutations for any given sample
        for _ in range(random.randint(0, len(alleles))):
            # pick a random mutation
            mut_i = random.randint(0, len(alleles) - 1)
            # calc 0 index of location
            mut_loc = allele_index[mut_i] - GEN_START
            # get possible variants that are not in the base genome
            alleles_i = list(set(alleles[mut_i]) - set([BASE_GENOME[mut_loc]]))
            # splice in a random allele
            full_seq = full_seq[:mut_loc] + random.choice(alleles_i) + full_seq[mut_loc + 1:]

        with open(f".{os.sep}{GENE}{os.sep}normal{os.sep}{i}.txt", 'w') as f:
            f.write(full_seq)
    # Forces one sample to be the base genome
    with open(f".{os.sep}{GENE}{os.sep}normal{os.sep}0.txt", 'w') as f:
        f.write(BASE_GENOME)


if __name__ == '__main__':
    GENE = sys.argv[1]

    # fetch base genome
    with open(f".{os.sep}{GENE}{os.sep}base_gene.txt", 'r') as f:
        text = f.read().strip().split('\n')
    # uses regex to search for gene start
    GEN_START = int(re.search('chromosome:\S+?:\S+?:(\S+?):', text[0]).groups()[0])
    BASE_GENOME = ''.join(text[1:])

    # reads ensembl variants from tsv file
    data = pandas.read_csv(f".{os.sep}{GENE}{os.sep}benign_vars.tsv", sep='\t')
    # ignores pandas complaining about not using regex matched groups
    warnings.filterwarnings('ignore', 'This pattern is interpreted as')
    # drops any rows where any variant has more than 2 bp (multi base variant), or 
    # is null (indel)
    VARS = data.drop(data[data['Variant alleles'].str.contains('([ACGT]{2,}|-)')].index)

    # get number of donor sequences (will generate an equal number of benign)
    num_samples = len(glob.glob(f".{os.sep}{GENE}{os.sep}cancer{os.sep}*.txt"))

    gen_genomes()
