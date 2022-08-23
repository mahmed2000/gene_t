import glob, json, re, os, sys

def parse_data():
    donor_list = set([i['donorId'] for i in raw_data])

    parsed_data = {}
    for donor in donor_list:
        parsed_data[donor] = []
    for mutation_data in raw_data:
        if mutation_data['start'] >= GEN_START and mutation_data['start'] <= GEN_END:
            parsed_data[mutation_data['donorId']] += [(mutation_data['start'], mutation_data['mutation'])]
    return parsed_data

def gen_donor_genome(mut_data):
    tmp = base_genome
    for mut in mut_data:
        mut_index = mut[0] - GEN_START
        allele = mut[1].split('>')[1]
        tmp = tmp[:mut_index] + allele + tmp[mut_index + 1:]
    return tmp

def gen_genomes():
    for donor in parsed_data.keys():
        donor_genome = gen_donor_genome(parsed_data[donor])
        with open(f"./{gene}/cancer/{donor}.txt", 'w') as f:
            f.write(donor_genome)

if __name__ == '__main__':
    gene = sys.argv[1]
    file_list = glob.glob(f"./{gene}/*.json")
    
    raw_data = []
    for file in file_list:
        with open(file, 'r') as f:
            raw_data += json.load(f)

    with open(f"./{gene}/base_gene.txt", 'r') as f:
        text = f.read().strip().split('\n')
    base_genome = ''.join(text[1:])
    GEN_START, GEN_END = list(map(int, re.search('chromosome:\S+?:\S+?:(\S+?):(\S+?):', text[0]).groups()))
    
    parsed_data = parse_data()
    gen_genomes()

