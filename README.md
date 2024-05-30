# Knowledge Graphs for Real World Rumour Verification

The PHEME dataset can be downloaded from https://drive.google.com/file/d/10fBgqLazWCNJkRx-jNucYZHAH6LFq9r6/view?usp=sharing

Run the code in the following order (the .pkl files generated save the output at each step of the process)

It is very likely that the directory name for the dataset will need changing to run the code

===

_1_googler (the articles retrieved will likely be different to ours)

_2_url_to_evidence (some URLs may dead)

1_data_to_pkl

2_entitiy_disambiguation

3_sentence_extractor

4_triples_nouns

5_kg_builder_silly

6_fixed_roberta_mha (is configured to run score/score/original, this can be changed around line 400)

