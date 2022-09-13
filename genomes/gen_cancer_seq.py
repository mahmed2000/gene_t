####################################################################################
#
#   Generates cancer genome sequences from a base genome from ensembl
#
#   Requires:
#       Correct directory structure as in README
#
#       Param: Directory to generate for (eg, TP53 in example branch)
#
####################################################################################

import glob, json, re, os, sys

def parse_data():
    # fetch all unique donors
    donor_list = set([i['donorId'] for i in raw_data])

    parsed_data = {}
    for donor in donor_list:
        parsed_data[donor] = []
    for mutation_data in raw_data:
        # check to make sure mutations are in range, breaks other scripts otherwise
        if mutation_data['start'] >= GEN_START and mutation_data['start'] <= GEN_END:
            parsed_data[mutation_data['donorId']] += [(mutation_data['start'], mutation_data['mutation'])]
    return parsed_data

def gen_donor_genome(mut_data):
    tmp = base_genome
    for mut in mut_data:
        # 0 indexed location of mutation from start of gene
        mut_index = mut[0] - GEN_START
        allele = mut[1].split('>')[1]   # eg gets T from A>T
        
        # splice and insert variant
        #
        # ...ACGCCGC>A<AACTCTC...
        # tmp[:i]    ^  tmp[i+1:]
        #            T
        tmp = tmp[:mut_index] + allele + tmp[mut_index + 1:]
    return tmp

def gen_genomes():
    os.mkdir(f".{os.sep}{gene}{os.sep}cancer{os.sep}")
    for donor in parsed_data.keys():
        donor_genome = gen_donor_genome(parsed_data[donor])
        with open(f".{os.sep}{gene}{os.sep}cancer{os.sep}{donor}.txt", 'w') as f:
            f.write(donor_genome)

if __name__ == '__main__':
    gene = sys.argv[1]

    # fetch all jsons downloaded from ICGC
    file_list = glob.glob(f".{os.sep}{gene}{os.sep}occur*.json")
    raw_data = []
    for file in file_list:
        with open(file, 'r') as f:
            raw_data += json.load(f)

    # read base genome from ensembl
    with open(f".{os.sep}{gene}{os.sep}base_gene.txt", 'r') as f:
        text = f.read().strip().split('\n')
    base_genome = ''.join(text[1:])
    # get start and end for gene using regex
    GEN_START, GEN_END = list(map(int, re.search('chromosome:\S+?:\S+?:(\S+?):(\S+?):', text[0]).groups()))
    
    parsed_data = parse_data()
    gen_genomes()

