---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:16000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 'Title: Slipstream | Plot: Slipstream is a 1989 post-apocalyptic
    science fiction adventure film. The plot has an emphasis on aviation and contains
    many common science-fiction themes, such as taking place in a dystopian future
    in which the landscape of the Earth itself has been changed and is windswept by
    storms of great power. There are also numerous sub-plots, such as free will and
    humanity amongst artificial intelligence. | Genres: Science Fiction, Adventure
    | Director: Steven Lisberger | Cast: Mark Hamill, Kitty Aldridge, Bob Peck, Bill
    Paxton, Eleanor David'
  sentences:
  - 'Title: American Grindhouse | Plot: This documentary explores the hidden history
    of the American Exploitation Film. The movie digs deep into this often overlooked
    category of U.S. cinema and unearths the shameless and occasionally shocking origins
    of this popular entertainment. | Genres: Documentary, History | Keywords: film
    director, exploitation, grindhouse, film production | Director: Elijah Drenner
    | Cast: Robert Forster, Allison Anders, John Landis, Kim Morgan, Fred Olen Ray
    | Studio: Lux Digital Pictures, End Films'
  - 'Title: Tron | Plot: As Kevin Flynn searches for proof that he invented a hit
    video game, he is "digitalized" by a laser and finds himself inside The Grid,
    where programs suffer under the tyrannical rule of the Master Control Program.
    With the help of TRON, a security program, Flynn seeks to free The Grid from the
    MCP. | Genres: Science Fiction, Action, Adventure | Keywords: video game, hacker,
    virtual reality, dystopia, computer, science fiction | Director: Steven Lisberger
    | Cast: Jeff Bridges, Bruce Boxleitner, David Warner, Cindy Morgan, Barnard Hughes
    | Studio: Walt Disney Pictures, Lisberger/Kushner'
  - 'Title: Born to Be Blue | Plot: Jazz legend Chet Baker finds love and redemption
    when he stars in a movie about his own troubled life to mount a comeback. | Genres:
    Drama, Music | Keywords: jazz, biography, historical figure | Director: Robert
    Budreau | Cast: Ethan Hawke, Carmen Ejogo, Callum Keith Rennie, Stephen McHattie,
    Janet-Laine Green | Studio: New Real Films, Lumanity Production'
- source_sentence: 'Title: Detective Story | Plot: Tells the story of one day in the
    lives of the various people who populate a police detective squad. An embittered
    cop, Det. Jim McLeod (Douglas), leads a precinct of characters in their grim daily
    battle with the city''s lowlife. The characters who pass through the precinct
    over the course of the day include a young petty embezzler, a pair of burglars,
    and a naive shoplifter. | Genres: Crime, Drama | Keywords: police station | Director:
    William Wyler | Cast: Kirk Douglas, Eleanor Parker, William Bendix, Cathy O''Donnell,
    George Macready | Studio: Paramount Pictures'
  sentences:
  - 'Title: Skin Game | Plot: Quincy Drew (Garner) and Jason O''Rourke (Gossett) travel
    from town to town in the south of the United States during the slavery era. Drew
    claims to be a down-on-his-luck slave owner who is selling O''Rourke as a slave.
    Quincy gets the bidding rolling, sells Jason, and the two later meet up to split
    the profit. Jason was born a free man in New Jersey and is very well educated.
    | Genres: Action, Comedy, Western | Keywords: con man, slave | Director: Gordon
    Douglas | Cast: James Garner, Louis Gossett, Jr., Susan Clark, Brenda Sykes, Ed
    Asner | Studio: Cherokee Productions, Warner Bros.'
  - 'Title: The Accidental Tourist | Plot: After the death of his son, Macon Leary,
    a travel writer, seems to be sleep walking through life. Macon''s wife, seems
    to be having trouble too, and thinks it would be best if the two would just split
    up. After the break up, Macon meets a strange outgoing woman, who seems to bring
    him back down to earth. After starting a relationship with the outgoing woman,
    Macon''s wife seems to think that their marriage is still worth a try. Macon is
    then forced to deal many decisions | Genres: Drama, Romance | Keywords: travel
    | Director: Lawrence Kasdan | Cast: William Hurt, Kathleen Turner, Geena Davis,
    Amy Wright, David Ogden Stiers | Studio: Warner Bros.'
  - 'Title: Tales of Terror | Plot: Three stories adapted from the work of Edgar Allen
    Poe:  1) A man and his daughter are reunited, but the blame for the death of his
    wife hangs over them, unresolved.  2) A derelict challenges the local wine-tasting
    champion to a competition, but finds the man''s attention to his wife worthy of
    more dramatic action.  3) A man dying and in great pain agrees to be hypnotized
    at the moment of death, with unexpected consequences. | Genres: Thriller, Comedy,
    Horror, Mystery | Keywords: cat, spider, reincarnation, vase, wine, buried alive,
    horror, anthology, murder, zombie, corpse, extramarital affair, drunk, hypnotist,
    tomb | Director: Roger Corman | Cast: Vincent Price, Peter Lorre, Basil Rathbone,
    Debra Paget, Joyce Jameson | Studio: Alta Vista Productions'
- source_sentence: 'Title: Summer Villa | Plot: Although a successful romance novelist,
    Terry Russell hasn''t had luck in her own love life. After a disastrous first
    date with cocky, hot-shot New York chef Matthew Everston, she retreats to her
    friend''s French villa for the summer to finish her latest novel, with her reluctant
    teenage daughter in tow. | Genres: Romance, Comedy | Director: Pat Kiely | Cast:
    Hilarie Burton, Victor Webster, Jocelin Haas, Cristina Rosato, Brittany Drisdelle
    | Studio: The Hallmark Channel, Marvista Entertainment, RG Productions'
  sentences:
  - 'Title: City Lights | Plot: City Lights is the first silent film that Charlie
    Chaplin directed after he established himself with sound accompanied films. The
    film is about a penniless man who falls in love with a flower girl. The film was
    a great success and today is deemed a cult classic. | Genres: Comedy, Drama, Romance
    | Keywords: suicide attempt, operation, blindness and impaired vision, love of
    one''s life, eye operation, flower shop, flower girl, tramp, love, millionaire,
    blind girl, little tramp | Director: Charlie Chaplin | Cast: Charlie Chaplin,
    Virginia Cherrill, Florence Lee, Harry Myers, Al Ernest Garcia | Studio: Charles
    Chaplin Productions'
  - 'Title: Blue Caprice | Plot: A narrative feature film inspired by the events known
    as the Beltway sniper attacks. | Genres: Crime, Drama, Mystery | Keywords: prison,
    sniper, biography, serial killer, lawyer, car | Director: Alexandre Moors | Cast:
    Isaiah Washington, Tequan Richmond, Tim Blake Nelson, Joey Lauren Adams, Leo Fitzpatrick
    | Studio: Intrinsic Value Films'
  - 'Title: Britannia Hospital | Plot: Britannia Hospital, an esteemed English institution,
    is marking its gala anniversary with a visit by the Queen Mother herself. But
    when investigative reporter Mick Travis arrives to cover the celebration, he finds
    the hospital under siege by striking workers, ruthless unions, violent demonstrators,
    racist aristocrats, and African cannibal dictator and sinister human experiments.
    | Genres: Horror, Comedy, Drama, Science Fiction | Director: Lindsay Anderson
    | Cast: Graham Crowden, Leonard Rossiter, Malcolm McDowell, Joan Plowright, Mark
    Hamill | Studio: EMI Films Ltd.'
- source_sentence: 'Title: The Aristocrats | Plot: One hundred superstar comedians
    tell the same very, VERY dirty, filthy joke--one shared privately by comics since
    Vaudeville. | Genres: Comedy, Documentary | Keywords: aftercreditsstinger, duringcreditsstinger
    | Director: Paul Provenza | Cast: Jason Alexander, Chris Albrecht, Hank Azaria,
    Shelley Berman, Steven Gary Banks | Studio: Mighty Cheese Productions'
  sentences:
  - 'Title: Surviving Picasso | Plot: The passionate Merchant-Ivory drama tells the
    story of Francoise Gilot, the only lover of Pablo Picasso who was strong enough
    to withstand his ferocious cruelty and move on with her life. | Genres: Drama,
    Romance | Keywords: paris, love triangle, female nudity, painter, infidelity,
    nudity, womanizer, mistress, pregnancy, older man younger woman relationship,
    physical abuse, picasso | Director: James Ivory | Cast: Anthony Hopkins, Natascha
    McElhone, Julianne Moore, Joss Ackland, Joan Plowright | Studio: Merchant Ivory
    Productions'
  - 'Title: Jackass 3D | Plot: Jackass 3D is a 3-D film and the third movie of the
    Jackass series. It follows the same premise as the first two movies, as well as
    the TV series. It is a compilation of various pranks, stunts and skits. Before
    the movie begins, a brief introduction is made by Beavis and Butt-head explaining
    the 3D technology behind the movie. The intro features the cast lining up and
    then being attacked by various objects in slow-motion. The movie marks the 10th
    anniversary of the franchise, started in 2000. | Genres: Comedy, Documentary,
    Action | Keywords: pain, stunts, stuntman, stupidity, comedy, duringcreditsstinger,
    3d | Director: Jeff Tremaine | Cast: Johnny Knoxville, Bam Margera, Ryan Dunn,
    Steve-O, Chris Pontius | Studio: MTV Films'
  - 'Title: George Carlin: Life Is Worth Losing | Plot: Carlin returns to the stage
    in his 13th live comedy stand-up special, performed at the Beacon Theatre in New
    York City for HBO®. His spot-on observations on the deterioration of human behavior
    include Americans’ obsession with their two favorite addictions - shopping and
    eating; his creative idea for The All-Suicide Channel, a new reality TV network;
    and the glorious rebirth of the planet to its original pristine condition - once
    the fires and floods destroy life as we know it. | Genres: Documentary, Comedy,
    TV Movie | Keywords: comedian, religion and supernatural, dying and death, concert,
    politics, made for cable tv, stand-up comedy, stand-up comedian, tv movie, tv
    special | Director: Rocco Urbisci | Cast: George Carlin | Studio: Cable Stuff
    Productions'
- source_sentence: 'Title: Runaway | Plot: Michael Adler has run away from his suburban
    home with his little brother Dylan. Hiding out in a quiet, rural town, Michael''s
    convinced he can make a better life for both of them. While Dylan stays holed
    up in a cheap motel all day, Michael works at a convenience store where everything
    starts to come together for him. But as Michael falls in love with his beautiful
    co-worker, Carly, his past begin | Genres: Drama, Thriller | Keywords: brother
    brother relationship, runaway, motel, love, co-worker, hiding, store, flashback,
    psychotherapist | Director: Tim McCann | Cast: Aaron Stanford, Robin Tunney, Peter
    Gerety, Melissa Leo, Terry Kinney | Studio: Filbert Steps Productions'
  sentences:
  - 'Title: King of the Belgians | Plot: The King of the Belgians is on a state visit
    in Istanbul when his country falls apart. He must return home at once to save
    his kingdom. But a solar storm causes airspace and communications to shut down.
    No planes. No phones. With the help of a British filmmaker and a troupe of Bulgarian
    folk singers, the King and his entourage manage to escape over the border. Incognito.
    Thus begins an odyssey across the Balkans during which the King discovers the
    real world - and himself. | Genres: Comedy, Drama | Keywords: road trip, woman
    director | Director: Peter Brosens | Cast: Peter van den Begin, Valentin Ganev,
    Goran Radaković, Lucie Debay, Titus De Voogdt | Studio: Art Fest, Bo Films, Entre
    chiens et loups'
  - 'Title: Phantasm II | Plot: Mike, after his release from a psychiatric hospital,
    teams up with his old pal Reggie to hunt down the Tall Man, who is at it again.
    A mysterious, beautiful girl has also become part of Mike''s dreams, and they
    must find her before the Tall Man does. | Genres: Action, Horror, Science Fiction,
    Thriller | Keywords: portal, undertaker, evil, tall man, sentinals | Director:
    Don Coscarelli | Cast: James Le Gros, Reggie Bannister, Angus Scrimm, Paula Irvine,
    Samantha Phillips | Studio: Universal Pictures, Spacegate Productions, Starway
    International Inc.'
  - 'Title: Pretty Ugly People | Plot: The grass might not be as green as one might
    think on the other side of obesity. Pretty Ugly People takes a wickedly comedic
    look at body image, self-loathing and sex within a group of estranged friends
    in their mid-30s. Lucy was obese her entire life and had always longed to be a
    thin so she could finally lose her virginity. | Genres: Comedy | Keywords: independent
    film, comedy | Director: Tate Taylor | Cast: Missi Pyle, Melissa McCarthy, Josh
    Hopkins, Octavia Spencer, Jack Noseworthy'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-mpnet-base-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: movie validation
      type: movie-validation
    metrics:
    - type: pearson_cosine
      value: 0.9637438140493771
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.895957970624571
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 384 tokens
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
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False, 'architecture': 'MPNetModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    "Title: Runaway | Plot: Michael Adler has run away from his suburban home with his little brother Dylan. Hiding out in a quiet, rural town, Michael's convinced he can make a better life for both of them. While Dylan stays holed up in a cheap motel all day, Michael works at a convenience store where everything starts to come together for him. But as Michael falls in love with his beautiful co-worker, Carly, his past begin | Genres: Drama, Thriller | Keywords: brother brother relationship, runaway, motel, love, co-worker, hiding, store, flashback, psychotherapist | Director: Tim McCann | Cast: Aaron Stanford, Robin Tunney, Peter Gerety, Melissa Leo, Terry Kinney | Studio: Filbert Steps Productions",
    'Title: Pretty Ugly People | Plot: The grass might not be as green as one might think on the other side of obesity. Pretty Ugly People takes a wickedly comedic look at body image, self-loathing and sex within a group of estranged friends in their mid-30s. Lucy was obese her entire life and had always longed to be a thin so she could finally lose her virginity. | Genres: Comedy | Keywords: independent film, comedy | Director: Tate Taylor | Cast: Missi Pyle, Melissa McCarthy, Josh Hopkins, Octavia Spencer, Jack Noseworthy',
    'Title: King of the Belgians | Plot: The King of the Belgians is on a state visit in Istanbul when his country falls apart. He must return home at once to save his kingdom. But a solar storm causes airspace and communications to shut down. No planes. No phones. With the help of a British filmmaker and a troupe of Bulgarian folk singers, the King and his entourage manage to escape over the border. Incognito. Thus begins an odyssey across the Balkans during which the King discovers the real world - and himself. | Genres: Comedy, Drama | Keywords: road trip, woman director | Director: Peter Brosens | Cast: Peter van den Begin, Valentin Ganev, Goran Radaković, Lucie Debay, Titus De Voogdt | Studio: Art Fest, Bo Films, Entre chiens et loups',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.0216, 0.5181],
#         [0.0216, 1.0000, 0.4188],
#         [0.5181, 0.4188, 1.0000]])
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

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `movie-validation`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value     |
|:--------------------|:----------|
| pearson_cosine      | 0.9637    |
| **spearman_cosine** | **0.896** |

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

* Size: 16,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                           | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                               | float                                                          |
  | details | <ul><li>min: 45 tokens</li><li>mean: 136.0 tokens</li><li>max: 294 tokens</li></ul> | <ul><li>min: 48 tokens</li><li>mean: 137.53 tokens</li><li>max: 294 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.45</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | label                           |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------|
  | <code>Title: Sandy Wexler \| Plot: Sandy Wexler (Adam Sandler) is a talent manager working in Los Angeles in the 1990s, diligently representing a group of eccentric clients on the fringes of show business. His single minded devotion is put to the test when he falls in love with his newest client, Courtney Clarke, a tremendously talented singer who he discovers at an amusement park. Over the course of a decade, the two of them play out a star-crossed love story. \| Genres: Comedy \| Keywords: show business, los angeles, talent manager, 1990s, eccentric clients \| Director: Steven Brill \| Cast: Adam Sandler, Kevin James, Terry Crews, Rob Schneider, Nick Swardson \| Studio: Happy Madison Productions, NetFlix</code>                                                                                                                                                             | <code>Title: Mr. Deeds \| Plot: When Longfellow Deeds, a small-town pizzeria owner and poet, inherits $40 billion from his deceased uncle, he quickly begins rolling in a different kind of dough. Moving to the big city, Deeds finds himself besieged by opportunists all gunning for their piece of the pie. Babe, a television tabloid reporter, poses as an innocent small-town girl to do an exposé on Deeds. \| Genres: Comedy, Romance \| Keywords: love letter, new hampshire, ferrari, liar, city country contrast, inheritance, billionaire, new york city, kindness, fable, apple tree, corporate take over, chrysler building \| Director: Steven Brill \| Cast: Adam Sandler, Winona Ryder, John Turturro, Allen Covert, Peter Gallagher \| Studio: New Line Cinema, Columbia Pictures Corporation, Out of the Blue... Entertainment</code> | <code>0.7999999999999999</code> |
  | <code>Title: Moon 44 \| Plot: Year 2038: The mineral resources of the earth are drained, in space there are fights for the last deposits on other planets and satellites. This is the situation when one of the bigger mining corporations has lost all but one mineral moons and many of their fully automatic mining robots are disappearing on their flight home. Since nobody else wants the job, they send prisoners to defend the mining station. Among them undercover agent Stone, who shall clear the whereabouts of the expensive robots. In an atmosphere of corruption, fear and hatred he gets between the fronts of rivaling groups. \| Genres: Science Fiction \| Keywords: raw materials, mondbasis, robot \| Director: Roland Emmerich \| Cast: Michael Paré, Lisa Eichhorn, Dean Devlin, Brian Thompson, Malcolm McDowell \| Studio: Centropolis Film Productions, Overseas FilmGroup</code> | <code>Title: Independence Day: Resurgence \| Plot: We always knew they were coming back. Using recovered alien technology, the nations of Earth have collaborated on an immense defense program to protect the planet. But nothing can prepare us for the aliens’ advanced and unprecedented force. Only the ingenuity of a few brave men and women can bring our world back from the brink of extinction. \| Genres: Action, Adventure, Science Fiction \| Keywords: alternate history, alien invasion \| Director: Roland Emmerich \| Cast: Liam Hemsworth, Jeff Goldblum, Bill Pullman, Maika Monroe, Sela Ward \| Studio: Twentieth Century Fox Film Corporation, Centropolis Entertainment, TSG Entertainment</code>                                                                                                                                 | <code>0.7</code>                |
  | <code>Title: Dracula 3D \| Plot: When Englishman Jonathan Harker visits the exotic castle of Count Dracula, he is entranced by the mysterious aristocrat. But upon learning that the count has sinister designs on his wife, Mina, Harker seeks help from vampire slayer Van Helsing. \| Genres: Horror, Romance, Thriller \| Keywords: dracula, gothic horror, evil, bite mark, 3d, vlad \| Director: Dario Argento \| Cast: Thomas Kretschmann, Asia Argento, Rutger Hauer, Marta Gastini, Unax Ugalde \| Studio: Enrique Cerezo Producciones Cinematográficas S.A., Film Export Group, Les Films de l'Astre</code>                                                                                                                                                                                                                                                                                          | <code>Title: Modern Vampires \| Plot: A vampire hunter in southern California discovers that his son has been murdered by a gang of the undead and thus goes on a quest for revenge. \| Genres: Action, Comedy, Horror, Romance, Thriller \| Keywords: vampire, dracula, vampire hunter, van helsing \| Director: Richard Elfman \| Cast: Casper Van Dien, Rod Steiger, Kim Cattrall, Natasha Lyonne, Natasha Gregson Wagner \| Studio: Muse Productions, Storm Entertainment, MUSE/Wyman</code>                                                                                                                                                                                                                                                                                                                                                          | <code>1.0</code>                |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
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
- `num_train_epochs`: 4
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

### Training Logs
| Epoch | Step | Training Loss | movie-validation_spearman_cosine |
|:-----:|:----:|:-------------:|:--------------------------------:|
| 0.1   | 400  | -             | 0.8675                           |
| 0.125 | 500  | 0.0327        | -                                |
| 0.2   | 800  | -             | 0.8960                           |


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