wget http://magnitude.plasticity.ai/word2vec+subword/GoogleNews-vectors-negative300.magnitude
mv GoogleNews-vectors-negative300.magnitude vectors.magnitude
export EMBEDDING_FILE='vectors.magnitude'
python main.py