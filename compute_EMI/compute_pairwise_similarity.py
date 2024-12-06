import numpy as np
import random
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util, models
from pathlib import Path
data_path = Path('data/')


def get_embeddings(text, model_name_or_path):
    model = SentenceTransformer(model_name_or_path)
    corpus_embeddings = model.encode(text, batch_size=1024, show_progress_bar=True, convert_to_tensor=True)
    assert len(corpus_embeddings) == len(text)
    return corpus_embeddings

def get_average_similarity_2decades(decade_model_path=None,
                                    dictionary_words=[], vocab=[], sample_random_words=False,
                                   decades=[]):
    
    similarity_matrix = pd.DataFrame(index=decades, columns=decades, dtype=float)
    
    # Loop through each unique pair of decades, loading temporal models
    for i, decade1 in enumerate(decades):
        for decade2 in decades[i+1:]:
                        
            # Load the models for each decade
            model1 = f'{decade_model_path}decade-{decade1}-model'
            model2 = f'{decade_model_path}decade-{decade2}-model'
            embeddings1 = get_embeddings(dictionary_words, model1)
            if sample_random_words and vocab:
                print('generating random pairs')
                random_words = random.sample(vocab, len(dictionary_words))
                embeddings2 = get_embeddings(random_words, model2)
            else:
                embeddings2 = get_embeddings(dictionary_words, model2)
            
            print(embeddings1.shape)
            sim = util.pairwise_cos_sim(embeddings1, embeddings2).cpu().numpy()
            print(sim.shape)
            avg_similarity = np.mean(sim)
            similarity_matrix.loc[decade1, decade2] = avg_similarity
            similarity_matrix.loc[decade2, decade1] = avg_similarity  # Symmetric entry           
            
    return similarity_matrix
with open(Path(data_path, "vocab_wo_dictionary_stopwords_downsampled.txt"), "r") as f:
    vocab_downsampled = [line.strip() for line in f]

evidence_words = pd.read_csv(Path(data_path, 'evidence_lexicon.csv'))
evidence_words = list(evidence_words['evidence_keywords'])
evidence_matrix_2decades_downsampled = get_average_similarity_2decades(decade_model_path=Path(data_path, 'temporal_embeddings/2decades_models_withcompass_downsampled/w2v-sentence-transformers/'), dictionary_words=evidence_words, decades=list(range(1879, 2000, 20)))

evidence_matrix_2decades_randompair_downsampled = get_average_similarity_2decades(decade_model_path==Path(data_path, 'temporal_embeddings/2decades_models_withcompass_downsampled/w2v-sentence-transformers/'), 
        dictionary_words=evidence_words, vocab=vocab_downsampled, sample_random_words=True,
        decades=list(range(1879, 2000, 20))
    )
evidence_matrix_2decades_downsampled.to_csv(Path(data_path, 'evidence_pairwise_sim_temporal.csv'))
evidence_matrix_2decades_randompair_downsampled.to_csv(Path(data_path, 'evidence_randompair_sim_temporal.csv'))

intuition_words = pd.read_csv(Path(data_path, 'intuition_lexicon.csv'))
intuition_words = list(intuition_words['intuition_keywords'])
intuition_matrix_2decades_downsampled = get_average_similarity_2decades(decade_model_path=Path(data_path, 'temporal_embeddings/2decades_models_withcompass_downsampled/w2v-sentence-transformers/'),
        dictionary_words=intuition_words, decades=list(range(1879, 2000, 20))
    )

intuition_matrix_2decades_randompair_downsample = get_average_similarity_2decades(decade_model_path=Path(data_path, 'temporal_embeddings/2decades_models_withcompass_downsampled/w2v-sentence-transformers/'),
        dictionary_words=intuition_words, vocab=vocab_downsampled, sample_random_words=True,
        decades=list(range(1879, 2000, 20))
    )

intuition_matrix_2decades_downsampled.to_csv(Path(data_path, 'intuition_pairwise_sim_temporal.csv'))
intuition_matrix_2decades_randompair_downsample.to_csv(Path(data_path, 'intuition_randompair_sim_temporal.csv'))
