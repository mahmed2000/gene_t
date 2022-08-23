# Requirements
---
All scripts must be run from the working directory they are found in. Will not run on Windows without rewriting file path strings.
## Python
- Pytorch
- Scikit-learn
- Pandas
- Xlib (optional)

## Optional programs
- xclip
- wmctrl
- any GUI browser

---
# Usage
---
## Download and setup
Note: Only point substitutions are supported, and must be selected for the ICGC download.
Pick a gene of interest and make a directory for it in the `genomes` folder. Browse for the gene on grch37.ensembl.org, note the start and end base pair values. Adjust the end value such that there are a total number of base pairs in the range divisible by 8.

Browse for this range on grch37.ensembl.org, export the data and (extract and) save it under the new directory with the name `base_gene.txt`. Query for the variants in this range with benign clinical significance in BioMart. Remove the gene info from attributes to ensure unique variations, only the start scaffold location and variant alleles columns are necessary. Check "Unique only", export as TSV. (Extract and) save it under the directory with the name `benign_vars.tsv`.

Navigate to dcc.icgc.org/search/m/o. Filter mutations by the adjusted base pair location range. Apply other filters as needed, then navigate back to mutations -> mutation occurences. Download the JSON data for each page.

### (Optional) Use `data_scrap.py` to download ICGC data
With an open browser window, execute `wmctrl -l` from a terminal, and note any unique section in the line corresponding to the browser ("Opera", "Mozilla", etc). Edit the file and replace the value of `BROWSER` with this section (case-sensitive). 

After selecting the filters, and navigating back to occurences, scroll down and select "show 50 rows". Copy the URL and replace the value of `URL` with this. Modify the end of this URL so that `..."size":50,"from":1...` becomes `..."size":{},"from":{}...`.

Modify `TOTAL` with the total number of occurences in the filtered search.

`DL_POS` will need to be modified if not using a maximized window on a 1080p monitor. To do so, first ensure the download JSON button is visible when loading the page when scrolled to the top of the page, then execute the following in a terminal, switch over to the browser window and wait a few seconds:
```
python -c 'import Xlib.display, time;time.sleep(3);print(Xlib.display.Display().screen().root.query_pointer())'
```
Replace `DL_POS` with `(root_x, root_y)` from the output.

Modify the keyboard shortcuts if they do not do what the comments describe.

Finally, run python `data_scrap.py` and wait for it to finish. Move all the downloaded JSON files to the directory

---
## Generating files
Run `python gen_cancer_seq.py DIR`, then `python gen_normal_seq.py DIR`, where DIR is the name of the directory with all of the downloaded files for a given region.
`cd ../ml` to move to the `ml` directory, and run `python data_prep.py` to generate tensors for training.

---
## Training
Running `python model.py` within `ml` will train a different model for each directory created.
