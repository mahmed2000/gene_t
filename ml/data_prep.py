####################################################################################
#
#   Encodes genomes to tensors for training
#
#   Required:
#       Packages: torch
#       Correct directory structure as in README
#
#
####################################################################################

import torch, glob, re, os

# Maps nucleotides to values 00 - 11 for encoding (arbitrary order)
BP_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def process_genome(index, file):
    with open(file, 'r') as f:
        genome = f.read().strip()

    # ACCCGGCGTTAGCT...
    #
    # i//8:
    #    0         1
    # ACCCGGCG TTAGCT...    -   0000000000000000 0000000000000000 ...
    # ^
    #  ^                    -   0000000000000001 0000000000000000 ...
    #   ^                   -   0000000000000101 0000000000000000 ...
    #    ^                  -   0000000000010101 0000000000000000 ...
    #     ^                 -   0000000001010110 0000000000000000 ...
    #                           .
    #                           .
    #                           .
    #        ^              -   0001010110100110 0000000000000000 ...
    #          ^            -   0001010110100110 0000000000000011 ...
    #                           .
    #                           .
    #                           .
    #
    # A C C C G G C G - 00 01 01 01 10 10 01 10
    # 0 1 1 1 2 2 1 2 -  0  1  1  1  2  2  1  2
    # Each bp, bit shift left 2, add bp equivalent value
    for i, bp in enumerate(genome):
        data[index][i // 8] = (data[index][i // 8] << 2) + BP_MAP.get(bp)

if __name__ == '__main__':
    # gets all genes from genomes folders
    gene_list = [i.split('/')[-2] for i in  glob.glob('../genomes/*/')]
    
    os.mkdir('./data/')
    for gene in gene_list:
        print(f"\nLoading gene: {gene}")

        # fetch base genome, and find start and end bp location
        with open(f"../genomes/{gene}/base_gene.txt", 'r') as f:
            tmp = f.read().strip().split('\n')[0]
        start, end = re.search(r'GRCh37:\S+?:(\S+?):(\S+?):', tmp).groups()
        
        # compute number of features needed to encode (8 bp per feature)
        n_features = (int(end) - int(start) + 1) // 8

        # fetch genome files
        cancer_files = glob.glob(f"../genomes/{gene}/cancer/*.txt")
        normal_files = glob.glob(f"../genomes/{gene}/normal/*.txt")

        n_samples = len(cancer_files) + len(normal_files)
        
        # int32 because uint16 not supported, int64 needed for labels
        data = torch.zeros((n_samples, n_features), dtype=torch.int32)
        labels = torch.zeros((n_samples), dtype=torch.int64)

        # sets last j labels to 1, (last j entries will be benign files)
        labels[len(cancer_files):] = 1

        for i, file in enumerate(cancer_files + normal_files):
            print(''*80, end='\r')
            print(f"Loading file \t{i} of \t{n_samples}", end='\r')
            process_genome(i, file)

        # saves data and labels in same pt file, by gene
        torch.save({'data': data, 'labels': labels}, f"data/{gene}.pt")
