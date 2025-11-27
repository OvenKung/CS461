---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:4624
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-base-en-v1.5
widget:
- source_sentence: Documentary movie with great story
  sentences:
  - Batkid Begins. On November 15, 2013, the world came together to grant one 5-year-old
    leukemia patient his wish to be Batman for a day. "Batkid Begins" looks at why
    and how this phenomenon took place, becoming one of the biggest "good news" stories
    of all time.
  - The Great Silence. A mute gunslinger fights in the defense of a group of outlaws
    and a vengeful young widow, against a group of ruthless bounty hunters.
  - The Whale. A reclusive English teacher suffering from severe obesity attempts
    to reconnect with his estranged teenage daughter for one last chance at redemption.
- source_sentence: movie directed by Jordan Barker
  sentences:
  - The Marsh. Writer Claire Holloway is troubled by nightmares of Rose Marsh Farm.
    She decides to vacation at the farm which unbeknown to her is haunted by the ghost
    of a little girl and a teenage boy. Claire enlists the help of Geoffry Hunt to
    help uncover a decade old tragedy.
  - Sharpe's Enemy. Portugal 1813. A band of deserters, including Sharpe's old enemy,
    Obadiah Hakeswill, have captured two women, one the wife of a high-ranking English
    officer, and are holding them hostage for ransom. Sharpe is given the 60th Rifles
    and a Rocket troop, as well as his majority to rescue the women. But while Sharpe
    may be able to deal with his old enemy, he has yet to face a newer threat, the
    French Major Duclos.
  - My Super Ex-Girlfriend. When New York architect Matt Saunders dumps his new girlfriend
    Jenny Johnson - a smart, sexy and reluctant superhero known as G-Girl - she uses
    her powers to make his life a living hell!
- source_sentence: Romance movie with great story
  sentences:
  - Tanu Weds Manu Returns. The film is a sequel to Tanu Weds Manu (2011), in which
    stars Kangana Ranaut (Tanu) and R. Madhavan (Manu) reprise their roles from the
    original. Tanu and Manu's marriage collapses. What happens when Manu meets Tanu's
    lookalike Kusum - and when Tanu returns? Kangana Ranaut also portrays the additional
    role of a Haryanvi athlete Kusum in the film.
  - The Seventh Seal. When disillusioned Swedish knight Antonius Block returns home
    from the Crusades to find his country in the grips of the Black Death, he challenges
    Death to a chess match for his life. Tormented by the belief that God does not
    exist, Block sets off on a journey, meeting up with traveling players Jof and
    his wife, Mia, and becoming determined to evade Death long enough to commit one
    redemptive act while he still lives.
  - Earth Girls Are Easy. In this musical comedy, Valerie is dealing with her philandering
    fiancé, Ted, when she finds that a trio of aliens have crashed their spaceship
    into her swimming pool. Once the furry beings are shaved at her girlfriend's salon,
    the women discover three handsome men underneath. After absorbing the native culture
    via television, the spacemen are ready to hit the dating scene in 1980s Los Angeles.
- source_sentence: movie directed by Joss Whedon
  sentences:
  - 'Avengers: Age of Ultron. When Tony Stark tries to jumpstart a dormant peacekeeping
    program, things go awry and Earth’s Mightiest Heroes are put to the ultimate test
    as the fate of the planet hangs in the balance. As the villainous Ultron emerges,
    it is up to The Avengers to stop him from enacting his terrible plans, and soon
    uneasy alliances and unexpected action pave the way for an epic and unique global
    adventure.'
  - 'House III: The Horror Show. Detective Lucas McCarthy finally apprehends "Meat
    Cleaver Max" and watches the electric chair execution from the audience. But killing
    Max Jenke only elevated him to another level of reality. Now Lucas'' family is
    under attack, his sanity in question, and his house haunted. Aided by a disreputable
    college professor, can Lucas reclaim his mind, house, and family? Features Lance
    Henriksen as the Lucas McCarthy and Brion James as Max Jenke. One of the few movies
    featuring these actors as main characters.'
  - Talaash. A cop, investigating the mysterious death of a filmstar, meets a sex-worker,
    while he faces some personal problems psychologically. The mystery connects these
    people in a way, that ultimately changes their lives.
- source_sentence: Animation movie with great story
  sentences:
  - 'LEGO DC Comics Super Heroes: Batman: Be-Leaguered. Superman wants Batman to join
    his new superhero team, but Batman prides himself on being a self-sufficient loner.'
  - Killer Crocodile. A group of environmentalists arrives at a faraway tropical delta
    where toxic waste is being dumped. However the water also hides a giant crocodile.
    The corrupt local officials don't help much either.
  - TEKKEN. In the year of 2039, after World Wars destroy much of the civilization
    as we know it, territories are no longer run by governments, but by corporations;
    the mightiest of which is the Mishima Zaibatsu. In order to placate the seething
    masses of this dystopia, Mishima sponsors Tekken, a tournament in which fighters
    battle until only one is left standing.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-base-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) <!-- at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Animation movie with great story',
    'LEGO DC Comics Super Heroes: Batman: Be-Leaguered. Superman wants Batman to join his new superhero team, but Batman prides himself on being a self-sufficient loner.',
    'TEKKEN. In the year of 2039, after World Wars destroy much of the civilization as we know it, territories are no longer run by governments, but by corporations; the mightiest of which is the Mishima Zaibatsu. In order to placate the seething masses of this dystopia, Mishima sponsors Tekken, a tournament in which fighters battle until only one is left standing.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5283, 0.4749],
#         [0.5283, 1.0000, 0.4446],
#         [0.4749, 0.4446, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,624 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                         |
  |:--------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 7.29 tokens</li><li>max: 14 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 74.3 tokens</li><li>max: 244 tokens</li></ul> |
* Samples:
  | sentence_0                                 | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>War movie with great story</code>    | <code>Red Tails. The story of the Tuskegee Airmen, the first African-American pilots to fly in a combat squadron during World War II.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
  | <code>movie directed by Dylan Avery</code> | <code>Loose Change: 2nd Edition. What if...September 11th was not a surprise attack on America, but rather, a cold and calculated genocide by our own government?We were told that the twin towers were hit by commercial jetliners and subsequently brought down by jet fuel. We were told that the Pentagon was hit by a Boeing 757. We were told that flight 93 crashed in Shanksville, Pennsylvania. We were told that nineteen Arabs from halfway across the globe, acting under orders from Osama Bin Laden, were responsible. What you will see here will prove without a shadow of a doubt that everything you know about 9/11 is a complete fabrication. Conspiracy theory? It's not a theory if you can prove it.Written and narrated by Dylan Avery, this film presents a rebuttal to the official version of the September 11, 2001 terrorist attacks and the 9/11 Commission Report.</code> |
  | <code>movie directed by David Hare</code>  | <code>Salting the Battlefield. David Hare concludes his trilogy of films about MI5 renegade Johnny Worricker with another fugue on power, secrets and the British establishment. Johnny Worricker goes on the run with Margot Tyrell across Europe, and with the net closing in, the former MI5 man knows his only chance of resolving his problems is to return home and confront prime minister Alec Beasley.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.9.6
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.8.0
- Accelerate: 1.10.1
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->