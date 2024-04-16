model_name_or_path="../../data/sentence-transformers-model/"

python compute_sbert_avg_lexicon.py --model_name_or_path ${model_name_or_path}\
	--input_file "../../data/combined_congress1879_till_2022_filtered_nonprocedural.csv.gzip"\
	--output_file "../../data/combined_congress1879_till_2022_filtered_chunk_ddr_len_binadj_congress_word2vec.csv.gzip" \
	--evidence_lexicon "../../data/evidence_lexicon.csv"\
	--intuition_lexicon "../../data/intuition_lexicon.csv"\
	--compression_type gzip --id_column "speech_id" --chunk_text



