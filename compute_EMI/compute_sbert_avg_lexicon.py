from sentence_transformers import SentenceTransformer, util, models
import matplotlib.pyplot as plt
import sys
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from scipy.stats import zscore
import string
from wordfreq import top_n_list
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=100 )


def config(parser):
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    parser.add_argument('--evidence_lexicon')
    parser.add_argument('--intuition_lexicon')
    parser.add_argument('--save_embeddings', action="store_true")
    parser.add_argument('--smoke_test', action="store_true")
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--compression_type', type=str, default='infer')
    parser.add_argument('--length_threshold', type=int, default=10)
    parser.add_argument('--tab_delimiter', action="store_true")
    parser.add_argument('--chunk_text', action="store_true")
    parser.add_argument('--min_chunk_length', type=int, default=50 )
    parser.add_argument('--max_chunk_length', type=int, default=150)
    parser.add_argument('--id_column', type=str, default="speech_id")
    return parser 

def get_embeddings(text, model_name_or_path):
    model = SentenceTransformer(model_name_or_path)
    #encode text in batches 
    corpus_embeddings = model.encode(text, batch_size=1024, show_progress_bar=True, convert_to_tensor=True)
    assert len(corpus_embeddings) == len(text)
    return corpus_embeddings

def length_adjustment_bin(df, length_column='length', minimum_length=10):
    bins = range(minimum_length, df[length_column].max()+10, 10)
    df[f'{length_column}_bin'] = pd.cut(df[length_column], bins=bins,)
    df['evidence_mean'] = df.groupby(f'{length_column}_bin')['avg_evidence_score'].transform('mean')
    df['evidence_adj'] = df['avg_evidence_score'] - df['evidence_mean']

    df['intuition_mean'] = df.groupby(f'{length_column}_bin')['avg_intuition_score'].transform('mean')
    df['intuition_adj'] = df['avg_intuition_score'] - df['intuition_mean']
    return df

def evidence_minus_intuition_score(df, evidence_column='evidence_adj', intuition_column='intuition_adj'):
    df[['evidence_z', 'intuition_z']] = \
    df[[evidence_column, intuition_column]].apply(zscore)
    df['evidence_minus_intuition_score'] = df['evidence_z'] - df['intuition_z']
    return df

top100 = top_n_list('en', 100)
def count_top100(s):
    return len([1 for w in s.split() if w in set(top100)])

def preprocess(df, args):
    #replace multiple occurrence of .
    df.text.replace(to_replace=r"\.\.+", value=" ", regex=True, inplace=True)
    df.text.replace(to_replace=r"\-\-+", value=" ", regex=True, inplace=True)
    df.text.replace(to_replace=r"__+", value=" ", regex=True, inplace=True)
    df.text.replace(to_replace=r"\*\*+", value=" ", regex=True, inplace=True)
    df.text.replace(to_replace=r"\s+", value=" ", regex=True, inplace=True)
    df['length'] = df.text.parallel_apply(lambda x: len(x.split()))
    df = df[df.length > args.length_threshold]
    print(len(df))

    df['tokens_top100'] = df.text.parallel_apply(count_top100)
    df['fraction_top100'] = df.tokens_top100 / df.length
    try:
        print('sample top100 ', df[(df.fraction_top100 < 0.05)].sample(10).text.tolist())
        df = df[~(df.fraction_top100 < 0.05)] 
        print(len(df))
    except:
        print('nothing to sample')
    
    if args.chunk_text:
        def chunk_by_length(x):
            max_chunk_length = args.max_chunk_length
            words = x.split()
            if len(words) > max_chunk_length:
                chunks = [words[i:i+max_chunk_length] for i in range(0, len(words), max_chunk_length)]
                last_chunk_length = len(chunks[-1])
                if len(chunks) > 1 and last_chunk_length < args.min_chunk_length:
                    chunks[-2] = chunks[-2] + chunks[-1]
                    del chunks[-1]
                chunked = [" ".join(chunk) for chunk in chunks]
            else:
                chunked = [" ".join(words)]
            return chunked 

        df['text'] = df.text.parallel_apply(chunk_by_length)
        df = df.explode("text", ignore_index=True)    
        df = df.drop_duplicates(subset=['text']+[f'{args.id_column}'])
        df['chunk_length'] = df.text.parallel_apply(lambda x: len(x.split()))
    return df

def main(args):
    delimiter = '\t' if args.tab_delimiter else None
    if args.smoke_test:
        df = pd.read_csv(args.input_file, nrows=100_000, compression=args.compression_type, delimiter=delimiter, dtype={'speech_id':object})
    else:
        df = pd.read_csv(args.input_file, compression=args.compression_type, delimiter=delimiter, dtype={'speech_id':object})
    #rename text column if different from text
    if args.text_column != 'text':
        df.rename(columns = {args.text_column:'text'}, inplace = True)
    df['text'] = df['text'].astype(str)
    df = df.drop_duplicates(subset=['text']+[f'{args.id_column}'])
    df['congress'] = (((df['year'] - 1789)/2)+1).astype('int')
    df['starting_year'] = 2*(df['congress'] - 1)+1789
    print('Before pre-processing:', len(df))
    df = preprocess(df, args)
    print('After pre-processing:', len(df[f'{args.id_column}'].unique()))
    evidence_sim = torch.Tensor()#.to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
    intuition_sim = torch.Tensor()#.to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
    chunk_size = 1_000_000
    list_df = [df[idx:idx+chunk_size] for idx in range(0, len(df), chunk_size)]
    for batch in tqdm(list_df):
        batch_text = batch['text']
        batch_text = list(batch_text)
        text_embeddings = get_embeddings(batch_text, args.model_name_or_path)       

        if args.save_embeddings:
            import pickle 
            output_fn = args.output_file.replace(".csv", ".pkl")
            with open(output_fn, "wb") as fout:
                pickle.dump({'text': all_text, 'embeddings': text_embeddings}, fout, protocol=pickle.HIGHEST_PROTOCOL)

        evidence_keywords = pd.read_csv(args.evidence_lexicon) 
        evidence_keywords = list(evidence_keywords['evidence_keywords'])  
        evidence_embeddings = get_embeddings(evidence_keywords, args.model_name_or_path)
        evidence_embeddings = torch.mean(evidence_embeddings, dim=0)

        intuition_keywords = pd.read_csv(args.intuition_lexicon) 
        intuition_keywords = list(intuition_keywords['intuition_keywords'])  
        intuition_embeddings = get_embeddings(intuition_keywords, args.model_name_or_path)
        intuition_embeddings = torch.mean(intuition_embeddings, dim=0)

        evidence_sim = torch.cat((evidence_sim, util.cos_sim(text_embeddings, evidence_embeddings).cpu()), 0)
        intuition_sim = torch.cat((intuition_sim, util.cos_sim(text_embeddings, intuition_embeddings).cpu()), 0)

    avg_evidence_score = np.average(evidence_sim.cpu().numpy(), axis=1)  
    avg_intuition_score = np.average(intuition_sim.cpu().numpy(), axis=1)  
    df['avg_evidence_score'] = avg_evidence_score
    df['avg_intuition_score'] = avg_intuition_score

    length_column = 'chunk_length' if args.chunk_text else 'length'
    df = length_adjustment_bin(df, length_column=length_column, minimum_length=args.length_threshold)
    df = evidence_minus_intuition_score(df, evidence_column='evidence_adj', intuition_column='intuition_adj') 
    print(df.evidence_minus_intuition_score.head())
    print(df.evidence_minus_intuition_score.tail())
    df.to_csv(args.output_file, index=False, compression=args.compression_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = parser.parse_args()
    main(args)

