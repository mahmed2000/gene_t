import pandas, glob, random, sys, re, warnings


def gen_genomes():
    alleles = list(VARS.sort_values(by='Chromosome/scaffold position start (bp)')['Variant alleles'].str.split('/'))
    allele_index = list(VARS.sort_values(by='Chromosome/scaffold position start (bp)')['Chromosome/scaffold position start (bp)'])

    for i in range(1, num_samples):
        full_seq = BASE_GENOME
        for _ in range(random.randint(0, len(alleles))):
            mut_i = random.randint(0, len(alleles) - 1)
            mut_loc = allele_index[mut_i] - GEN_START
            alleles_i = list(set(alleles[mut_i]) - set([BASE_GENOME[mut_loc]]))
            full_seq = full_seq[:mut_loc] + random.choice(alleles_i) + full_seq[mut_loc + 1:]

        with open(f"./{GENE}/normal/{i}.txt", 'w') as f:
            f.write(full_seq)
    with open(f"./{GENE}/normal/0.txt", 'w') as f:
        f.write(BASE_GENOME)


if __name__ == '__main__':
    GENE = sys.argv[1]
    with open(f"./{GENE}/base_gene.txt", 'r') as f:
        text = f.read().strip().split('\n')
    GEN_START = int(re.search('chromosome:\S+?:\S+?:(\S+?):', text[0]).groups()[0])
    BASE_GENOME = ''.join(text[1:])

    data = pandas.read_csv(f"./{GENE}/benign_vars.tsv", sep='\t')
    warnings.filterwarnings('ignore', 'This pattern is interpreted as')
    VARS = data.drop(data[data['Variant alleles'].str.contains('([ACGT]{2,}|-)')].index)

    num_samples = len(glob.glob(f"./{GENE}/cancer/*.txt"))

    gen_genomes()
