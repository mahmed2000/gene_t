import torch, glob, re

BP_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def process_genome(index, file):
    with open(file, 'r') as f:
        genome = f.read().strip()
    for i, bp in enumerate(genome):
        data[index][i // 8] = (data[index][i // 8] << 2) + BP_MAP.get(bp)

if __name__ == '__main__':
    gene_list = [i.split('/')[-2] for i in  glob.glob('../genomes/*/')]

    for gene in gene_list:
        print(f"\nLoading gene: {gene}")
        with open(f"../genomes/{gene}/base_gene.txt", 'r') as f:
            tmp = f.read().strip().split('\n')[0]
        start, end = re.search(r'GRCh37:\S+?:(\S+?):(\S+?):', tmp).groups()
        n_features = (int(end) - int(start) + 1) // 8

        cancer_files = glob.glob(f"../genomes/{gene}/cancer/*.txt")
        normal_files = glob.glob(f"../genomes/{gene}/normal/*.txt")

        n_samples = len(cancer_files) + len(normal_files)
        
        data = torch.zeros((n_samples, n_features), dtype=torch.int32)
        labels = torch.zeros((n_samples), dtype=torch.int64)
        labels[len(cancer_files):] = 1

        for i, file in enumerate(cancer_files + normal_files):
            print(''*30, end='\r')
            print(f"Loading file \t{i} of \t{n_samples}", end='\r')
            process_genome(i, file)

        torch.save({'data': data, 'labels': labels}, f"data/{gene}.pt")
