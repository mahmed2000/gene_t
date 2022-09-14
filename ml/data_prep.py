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

import torch, glob, re, os, time, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    # Same for sliding window, too much work to draw
    for i, bp in enumerate(genome):
        # computes which feature (indexes) a bp is part of
        features_changed_index = range(max(0, i//SLIDER - FEAT_SIZE//SLIDER + 1), min(i//SLIDER + 1, n_features))
        for feat in features_changed_index:
            data[index][feat] = (data[index][feat] << 2) + BP_MAP.get(bp)

if __name__ == '__main__':
    # gets all genes from genomes folders
    gene_list = [i.split(os.sep)[-2] for i in glob.glob(f"..{os.sep}genomes{os.sep}*{os.sep}")]
    
    with open(f".{os.sep}gen_conf.json", 'r') as f:
        gen_confs = json.loads(f.read())

    os.makedirs(f".{os.sep}data{os.sep}", exist_ok = True)
    for gene in gene_list:
        print(f"\nLoading gene: {gene}")

        # fetch base genome, and find start and end bp location
        with open(f"..{os.sep}genomes{os.sep}{gene}{os.sep}base_gene.txt", 'r') as f:
            tmp = f.read().strip().split('\n')[0]
        start, end = re.search(r'GRCh37:\S+?:(\S+?):(\S+?):', tmp).groups()
        
        # compute number of features needed to encode (8 bp per feature, sliding window)
        n_bp = int(end) - int(start) + 1

        # fetch genome files
        cancer_files = glob.glob(f"..{os.sep}genomes{os.sep}{gene}{os.sep}cancer{os.sep}*.txt")
        normal_files = glob.glob(f"..{os.sep}genomes{os.sep}{gene}{os.sep}normal{os.sep}*.txt")

        n_samples = len(cancer_files) + len(normal_files)
        
        for gen_i, gen_conf in enumerate(gen_confs):
            print(f"\nConfig {gen_i + 1} of {len(gen_confs)}")
            FEAT_SIZE = gen_conf.get('FEAT_SIZE', 8)
            SLIDER = gen_conf.get('SLIDER', 8)

            n_features = (n_bp // FEAT_SIZE) + (n_bp // FEAT_SIZE - 1) * (FEAT_SIZE // SLIDER - 1)
            # int32 because uint16 not supported, int64 needed for labels
            data = torch.zeros((n_samples, n_features), dtype=torch.int32)
            labels = torch.zeros((n_samples), dtype=torch.int64)

            # sets last j labels to 1, (last j entries will be cancer files)
            labels[len(normal_files):] = 1

            st = time.time()

            for i, file in enumerate(normal_files + cancer_files):
                try:
                    eta = int(((n_samples - i) / (i / (time.time() - st))) // 60)
                except:
                    eta = 'N/A'

                print(' '*80, end='\r')
                print(f"Loading file \t{i} of \t{n_samples}\t| ETA\t{eta} min", end='\r')
                process_genome(i, file)
            data = torch.tensor(MinMaxScaler().fit_transform(data))
            data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.1, shuffle = True)

            # saves data and labels in same pt file, by gene
            torch.save({'data': data_train, 'labels': labels_train}, f"data{os.sep}{gene}_{FEAT_SIZE}_{SLIDER}_train.pt")
            torch.save({'data': data_test, 'labels': labels_test}, f"data{os.sep}{gene}_{FEAT_SIZE}_{SLIDER}_test.pt")
