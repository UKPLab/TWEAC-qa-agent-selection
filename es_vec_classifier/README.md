# Classification with Embeddings + kNN
We use Elasticsearch with a [plugin](https://github.com/lior-k/fast-elasticsearch-vector-scoring) for embedding-based kNN classification for the agent selection task.

## Usage
### 1. Fill Elasticsearch server with 'training' data for different agents
Create and fill the index with `python ../elasticsearch/create_index_with_embedding.py ../elasticsearch/config.json --new`.
Adapt [the config](../elasticsearch/config.json) with your choice of agents and adapt the paths to the data.

If you want to use docker, then you can [build the image with our Dockerfile](../elasticsearch/Dockerfile) (including the plugin) and start the server with `es_docker_plugin.sh` beforehand.

### 2. Evaluation
Run `python es_vec_classifier.py $config.json --eval` to evaluate with the given config.
See [our config](config.json) for an example.
The names for the agents have to be identical to those used during index creation.
In general, you only change the path from the training to the test folders.

#### Parameters:
**sentence_transformer_model**   
Either a URL for a Universal-Sentence-Encoder model (e.g. https://tfhub.dev/google/universal-sentence-encoder-qa/3)
or the name for a [sentence-transformer model](https://github.com/UKPLab/sentence-transformers).

**use_cosine**   
If true, use cosine for similarity, otherwise use dot product.

**k**   
The number of retrieved neighbors which make up the votes for the weighted voting for the classification.

**weighting**   
``uniform`` : each vote is weighted the same (1/k)  
``score``: each vote is weighted proportional to the score given by Elasticsearch

**class_weighting**   
If true, each vote is normalized by the number of examples for the agent in the training data