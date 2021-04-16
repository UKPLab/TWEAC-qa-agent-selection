#TWEAC: Transformer with Extendable QA Agent Classifiers
We use a easily extendable Transformer model which has a dedicated classification
head for each agent.
This makes adding (or removing) new agents as easy as adding a new head.

## Training & Evaluation
Run ```python run_transformer.py $config.yaml``` to train and/ or test a model.
See [configs/example_config.yaml](configs/example_config.yaml) for an example config detailing the parameters
and see [configs/standard_train_eval.yaml](configs/standard_train_eval.yaml) for a config to train & evaluate a model on 10 agents.


### Extending
To extend a model, you will need to update `all_agents` in the config along with
setting `agents_extended: 1` and providing a `base_model` which should be extended.
If you want to extend an extended model, you will first have to merge the old classification head with the extended one - see `merge_extended` in [iterative_add.py](iterative_add.py) for details.

We provide our code for iterative extension (with optional down-sampling of training examples for not-extended agents to achieve constant training
time regardless of agents - see half-and-half sampling in our publication for more details).

We also provide code for our leave-one-out evaluation setup.


#### Iterative Extension
We iteratively extend a TWEAK model initially trained with some agents one by one with multiple agents.
Run `python iterative_add.py $config --data_config=$data_config --stop=90 --start=0`.
See [configs/iterative_extend_reddit.yaml](configs/iterative_extend_reddit.yaml) for an example config for reddit
and see [../data/dataconfig_reddit.yaml](../data/dataconfig_reddit.yaml) for an example data config.

Note, you will have to remove the first 10 agents from the data config if you want to use it with the config for iterative extension
and you will have to train a TWEAK model with those initial 10 agents (see above on how to train your model).

#### Leave-one-out
We evaluate extension in a leave-one-out approach where given k agents, 
we train models with k-1 agents and extend with the remaining one, repeating this for all k agents.

Run `python leave_one_out.py --modes "finetune,extend" --econfig=$extend_config --fconfig=$finetune_config`
For the finetune config, you can use a standard training config (e.g. [configs/standard_train_eval.yaml](configs/standard_train_eval.yaml)).
Note, the folders for the finetuned models will be placed in folders in `out_dir` and will contain the left-out agent.
For the extend config, see [configs/loo_extend.yaml](configs/loo_extend.yaml).
Note, `base_model` has to point `out_dir` of the finetune config.