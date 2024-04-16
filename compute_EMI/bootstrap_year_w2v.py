import numpy as np
import pandas as pd
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from tqdm import tqdm
from pathlib import Path
import argparse

def config(parser):
    parser.add_argument('--input_file', type=str, default='../../data/combined_congress1879_till_2022_filtered_metadata_ddr_adj_chunk_filter_len_binadj_congress_word2vec.csv.gzip')
    parser.add_argument('--output_path', type=str, default='../../data/bootstrap_year_w2v/')
    return parser 

def main(args):
    chunked_filtered_df = pd.read_csv(args.input_file,
                           compression='gzip',
                           dtype={'speech_id':object},
                          )
    chunked_filtered_df = chunked_filtered_df[chunked_filtered_df.chamber.isin(['H', 'S'])]
    chunked_filtered_df['congress'] = (((chunked_filtered_df['year'] - 1789)/2)+1).astype('int')
    chunked_filtered_df['starting_year'] = 2*(chunked_filtered_df['congress'] - 1)+1789
    grouped_df = chunked_filtered_df.groupby('starting_year')

    for name, group in tqdm(grouped_df):
        df = pd.DataFrame()
        bs_result = bs.bootstrap(group.evidence_minus_intuition_score.values, stat_func=bs_stats.mean,
                                 num_iterations=10_000, num_threads=100,
                                 return_distribution=True
                                )
        df = pd.DataFrame(bs_result, columns=["evidence_minus_intuition_score"])
        df['starting_year'] = name
        df.to_csv(Path(args.output_path, f'{name}.csv'), index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = parser.parse_args()
    main(args)



