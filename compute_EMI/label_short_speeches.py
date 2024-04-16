import numpy as np
import pandas as pd
from statistics import mode
import ktrain
import string
import re
import sys
from pathlib import Path
import argparse
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=100)

def config(parser):
    parser.add_argument('--input_file', type=str, default='../../data/speeches_original/combined_congress1879_till_2022.csv.gzip')
    parser.add_argument('--output_path', type=str, default='../../data/speeches_original/')
    parser.add_argument('--nbsvm_path', type=str, default='../../data/procedural_model/nbsvm_train_full_length_heuristics_nv/')
    parser.add_argument('--fasttext_path', type=str, default='../../data/procedural_model/fasttext_train_full_length_heuristics_nv/')
    parser.add_argument('--logreg_path', type=str, default='../../data/procedural_model/logreg_train_full_length_heuristics_nv/')
    return parser 
  
replace_punct = re.compile('[%s]' % re.escape(string.punctuation))
def simplify_text(text):
    # drop all punctuation
    text = replace_punct.sub('', text)
    # lower case the text
    text = text.strip().lower()
    # convert all white space spans to single spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def main(args):
    combined = pd.read_csv(args.input_file, compression='gzip')
    combined['text'] = combined['text'].astype(str)
    combined['speech_id'] = combined['speech_id'].astype(str)
    combined['text'] = combined.text.parallel_apply(simplify_text)
    combined['token_count'] = combined.text.parallel_apply(lambda x: len(x.split()))
    combined['character_count'] = combined.text.parallel_apply(lambda x: len(x))
    
    #filter token_count > 2 & character_count > 15
    very_short_speeches = combined[(combined.token_count <= 2) | (combined.character_count <= 15)]
    short_speeches = combined[(combined.token_count > 2) & (combined.character_count > 15)]
    
    #subset character count < 400
    short_speeches = short_speeches[short_speeches.character_count < 400]
    
    def procedural_prediction(text, model_path):
        predictor = ktrain.load_predictor(model_path)
        predictions = predictor.predict(text)
        return predictions

    short_speeches['nbsvm'] = procedural_prediction(short_speeches.text.tolist(), args.nbsvm_path)
    short_speeches['fasttext'] = procedural_prediction(short_speeches.text.tolist(), args.fasttext_path)
    short_speeches['logreg'] = procedural_prediction(short_speeches.text.tolist(), args.logreg_path)
    short_speeches['combined_predictions'] = short_speeches[['nbsvm', 'fasttext', 'logreg']].values.tolist()
    short_speeches['ensemble_prediction'] = short_speeches['combined_predictions'].apply(lambda x: mode(x)) 
    
    procedural_speeches = short_speeches[short_speeches['ensemble_prediction'] == 'procedural']
    procedural_speeches = pd.concat([very_short_speeches, procedural_speeches])
    condition = (~combined['speech_id'].isin(procedural_speeches['speech_id'])) 
    combined = combined[condition]
    combined[['speech_id']].to_csv(Path(args.output_path, 'US_congress_1879till_12_2022_nonprocedural_speechids.csv.gzip'), index=False, compression='gzip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = parser.parse_args()
    main(args)

