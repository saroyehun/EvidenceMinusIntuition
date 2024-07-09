# Computational analysis of US congressional speeches reveals a shift from evidence to intuition

This repository contains the code for the analysis and results in the manuscript **[Computational analysis of US congressional speeches reveals a shift from evidence to intuition](https://arxiv.org/abs/2405.07323)**


## Notes
* We developed the codes in this repository with Python (3.6.13) and R(4.1.3) on Ubuntu 20.04.
* The scripts are numbered in the order in which the results in the paper are presented with results saved in the `output` directory.
* The aggregated EMI score and data for variables required to make plots and run statistical analysis are in this repository under the `data` directory.
* For the code in the directory `compute_EMI` to execute, the required Congressional speeches and embedding model are in a separate [OSF repository](https://doi.org/10.17605/OSF.IO/Z6UTW), because of their size.
    * The Python package dependencies for the scripts in the directory `compute_EMI` can be installed using the `requirements.txt` file i.e., `pip install -r requirements.txt`.
    * To compute the **EMI score** on the Congressional speeches, run the script `label_filtered_uscongress_congress_word2vec.sh` from within the `compute_EMI` directory.

  **For questions or clarifications please contact:**
  * Segun Aroyehun (segun.aroyehun@uni-konstnaz.de)
  *  David Garcia (david.garcia@uni-konstanz.de)
 
## Citation

```bibtex
@article{aroyehun2024computational,
  title={Computational analysis of US Congressional speeches reveals a shift from evidence to intuition},
  author={Aroyehun, Segun Taofeek and Simchon, Almog and Carrella, Fabio and Lasser, Jana and Lewandowsky, Stephan and Garcia, David},
  journal={arXiv preprint arXiv:2405.07323},
  year={2024}
```
  
