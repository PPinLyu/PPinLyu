```python
import numpy as np
import pandas as pd
import chardet
from collections import Counter

import re
import nltk
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
```

### Import Article 


```python
file_path = "/Users/pin.lyu/Documents/BC_Folder/NLP/Data/Scientists say they are close to resurrecting a lost species.txt"

# Detect the file encoding

with open(file_path, "rb") as file:
    
    raw_data = file.read()
    
    result = chardet.detect(raw_data)
    
    encoding = result["encoding"]

print(f"Detected encoding: {encoding}")
```

    Detected encoding: UTF-8-SIG



```python
# Use the detected encoding to load the file
with open(file_path, "r", encoding=encoding) as file:
    article = file.read()

print(article)  # Display the content of the file
```

    Scientists say they are close to resurrecting a lost species. Is the age of de-extinction upon us?
    The age of de-extinction may soon be a reality.
    Advances in genetic engineering and synthetic biology are making resurrecting animals once lost to this world a tangible prospect. The organizations and companies at the forefront of de-extinction efforts are promising success — and surprisingly soon.
    Those efforts just got a boost. Colossal Biosciences, the biotech company behind plans to revive the woolly mammoth, dodo and Tasmanian tiger, announced Wednesday it has raised an additional $200 million in investment, bringing its total funding to $435 million. That hefty sum grew from an initial $15 million in 2021 when entrepreneur Ben Lamm and Harvard University geneticist George Church founded the Dallas-based company.
    Within a decade or less, the world may see approximations of creatures only known from black-and-white photographs, taxidermy museum exhibits, and fossilized skeletons, with the ultimate goal of restoring the fauna to their natural habitat.
    Advocates say resurrecting extinct animals is attracting new investors with deep pockets to conservation. The scientific field pushes the boundaries of biotechnology in a way that will make it possible to save other species on the brink and offers a promising way to better protect and preserve present-day ecosystems, ultimately making them more resilient to the climate crisis.
    Skeptics, however, argue the efforts are an underscrutinized pet project of millionaires, whose money could be spent more effectively elsewhere. Detractors also assert that scientists will only ever be able to engineer unsatisfactory imitations of extinct animals. Raising and breeding such creatures, some experts warn, could imperil living animals used as surrogates and the ecosystems into which resurrected individuals might ultimately be released.
    “Who doesn’t want to see a dodo? Good God, I do. A mammoth. I mean, wow, amazing,” said Melanie Challenger, the deputy co-chair of the Nuffield Council on Bioethics in the United Kingdom.
    Challenger, who is the author of “How To Be Animal: A New History of What it Means To Be Human,” argues that de-extinction is a fundamentally misleading term. “It’s not de-extinction, it’s genetically engineering a novel organism to fulfill the functions, theoretically, of an extant (living) organism. You’re not bringing anything back from the dead,” she said. “And all the way through the process, there are different, quite gnarly ethical considerations.”
    Is de-extinction really possible?
    Scientists are pioneering and refining three techniques in their attempts to revive lost and rare species: cloning, genetic engineering, and traditional back breeding, a form of selective breeding that seeks to recreate lost traits from extinct species.
    From this resurrection tool kit, cloning has the capability to create an animal that’s almost genetically identical. Dolly the Sheep became the first cloned mammal almost 30 years ago, and recently scientists successfully cloned the endangered black-footed ferret. But the process has been hit-and-miss, and it’s unlikely to be useful in attempts to revive animals that disappeared a long time ago.
    Netherlands-based Grazelands Rewilding breeds a modern-day equivalent of the aurochs, an ox that features in prehistoric cave paintings. The giant animal disappeared from the wild in the 17th century. Aiming to restore wild landscapes in Europe, the group uses old-fashioned breeding methods, combined with some genetic knowledge, to identify the aurochs’ traits in living descendants: domesticated cattle.
    Now on the seventh generation, the tauros cattle, as they have been named, are more than 99% genetically similar to the extinct aurochs, said Ronald Goderie, the project’s managing director. The animals display physical changes, such as a darker coat color, and behavioral changes, such as how they respond to predators like wolves, over time.
    The scientists at Colossal are behind the most ambitious projects. This team wants to resurrect the mammoth, the flightless dodo and Tasmanian tiger, an Australian marsupial that went extinct in 1936. Colossal plans to recreate these creatures by editing the genome of the extinct animal’s closest living relative to make a hybrid animal that would be visually indistinguishable from its extinct forerunner. For the mammoth, that animal is the Asian elephant.
    High-profile investors in the endeavor include “Lord of the Rings” director Peter Jackson, socialite Paris Hilton, former professional football player Tom Brady and professional golfer Tiger Woods; as well as investment firms such as Breyer Capital. The latest infusion of cash comes from TWG Global, the investment vehicle of Mark Walter, controlling owner of the Los Angeles Dodgers baseball team and a co-owner of Chelsea Football Club in the UK.
    How close are scientists to reviving lost species?
    With the influx of capital, Lamm said the Colossal team may add another extinct animal to the to-do list as it makes headway in its three flagship projects.
    Recent milestones include creating the first induced pluripotent stem cells, or iPSCs, for Asian elephants. This special type of cell can be engineered in the lab to grow into any kind of elephant cell. It’s an important tool as researchers model, test and refine the scores of genetic changes they need to make to give an Asian elephant the traits of a mammoth needed for survival in a cold climate.
    For the Tasmanian tiger or thylacine, Lamm said the pace of progress has been quicker than expected. Colossal scientists have been able to make 300 genetic edits into a cell line of a fat-tailed dunnart, which is the marsupial that Colossal has chosen as its base species and future surrogate. The company has sequenced what Lamm described as the highest-quality ancient genome to date for any animal.
    The dodo is proving most challenging, Lamm said. Colossal has established a flock of Nicobar pigeons, the dodo’s closest living relative, which will act as donors for primordial germ cells that will be genetically edited to have dodo characteristics.
    However, many of the developments have not been published in scientific journals, meaning they can’t be scrutinized by other scientists as is typical during the peer-review process and won’t become publicly available for the benefit of the research community.
    Lamm said Colossal’s mission as a business is not to publish scientific papers, which is a monthslong, if not yearslong, process. However, he said that a paper on the creation of elephant iPSCs is in peer review. The company’s academic partners are planning to submit their work to journals, including the thylacine genome, in time, he added.
    Colossal has recruited respected high-profile scientists, and many other experts act in advisory roles, including some initially skeptical of some of the company’s goals. They include molecular paleobiologist Beth Shapiro, Colossal’s chief science officer, who is currently on a leave of absence from her role as professor of ecology and evolutionary biology at the University of California Santa Cruz.
    Shapiro is clear that de-extinction is not a solution to the extinction crisis, but she believes that the biotech tools she and her teams develop along the way can be applied more widely to protect and restore endangered species and ecosystems.
    “To be clear getting something that is 100% identical behaviorally, physiologically, genetically to a mammoth isn’t possible,” she told CNN in October. “Once a species is lost, it’s gone, and we need to be investing in making sure that things don’t become extinct.”
    Colossal is increasingly using its deep pockets to fund conservation efforts including work to save the world’s most endangered rhino species: the northern white rhino. The company is also collaborating on the development of a vaccine for a herpes-like disease that can kill elephants. And Colossal has entered into a partnership with the conservation organization Re:wild to use biotechnology in its projects.
    
    End goals
    Colossal’s stated end goal for its mammoth project is a world where the elephant-mammoth hybrids lumber through the Arctic permafrost compressing the snow and grass that insulates the ground, slowing down permafrost thaw and the release of carbon contained in this fragile ecosystem.
    It’s “absurd” to imagine herds of cold-adapted elephants making a significant impact on a region that’s warming faster than anywhere else in the world in the time frame necessary to make a difference in the climate crisis, said Christopher Preston, a professor of environmental philosophy at the University of Montana.
    Nevertheless, restoring lost species to fragile ecosystems has merit as a concept, added Preston, who is also author of “Tenacious Beasts: Wildlife Recoveries That Change How We Think About Animals.” He said he was impressed by Grazelands Rewilding’s tauros project, which he visited in the course of his work. The grazing habits of the hundreds of tauros cattle, herds of which also now live in parts of Spain, Czech Republic, Croatia and Romania, play a role in recreating an open landscape where other species can thrive.
    However, Clare Palmer, a professor of philosophy at Texas A&M University who specializes in animal and environmental ethics, noted that ecosystems are changing rapidly. She said bringing back animals may not work if the landscape is no longer the same.
    “We also don’t have good knowledge of the welfare needs of members of extinct species and the offspring, for instance, would not be taught by their parents how to hunt, or forage, or relate to other species members,” Palmer said.
    


#### EDA


```python
# Split the text into words

words = article.split()

# Count the number of words

word_count = len(words)

print(f"Number of words: {word_count}")
```

    Number of words: 1516



```python
# Count frequencies

word_freq = Counter(words)

# Most frequent words

print("Most Frequent 20 Words:")

for word, freq in word_freq.most_common(20):
    
    print(f"{word}: {freq}")

# Least frequent words

print("\nLeast Frequent 20 Words:")

for word, freq in word_freq.most_common()[:-21:-1]:
    
    print(f"{word}: {freq}")
```

    Most Frequent 20 Words:
    the: 89
    of: 51
    to: 47
    a: 39
    and: 33
    in: 32
    is: 20
    that: 19
    as: 16
    be: 13
    The: 12
    are: 11
    Colossal: 10
    has: 10
    an: 10
    extinct: 9
    said: 9
    from: 8
    species: 8
    not: 8
    
    Least Frequent 20 Words:
    Palmer: 1
    members,”: 1
    relate: 1
    forage,: 1
    hunt,: 1
    parents: 1
    taught: 1
    instance,: 1
    offspring,: 1
    members: 1
    needs: 1
    welfare: 1
    knowledge: 1
    good: 1
    “We: 1
    same.: 1
    longer: 1
    no: 1
    She: 1
    rapidly.: 1


### Summarization

#### Extractive Summarization


```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # Latent Semantic Analysis summarizer
```


```python
def summarize_text(text, num_sentences=5):
    
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    summarizer = LsaSummarizer()
    
    summary = summarizer(parser.document, num_sentences)
    
    return " ".join(str(sentence) for sentence in summary)

summary = summarize_text(article, num_sentences=5)

print(summary)
```

    Advances in genetic engineering and synthetic biology are making resurrecting animals once lost to this world a tangible prospect. That hefty sum grew from an initial $15 million in 2021 when entrepreneur Ben Lamm and Harvard University geneticist George Church founded the Dallas-based company. Skeptics, however, argue the efforts are an underscrutinized pet project of millionaires, whose money could be spent more effectively elsewhere. From this resurrection tool kit, cloning has the capability to create an animal that’s almost genetically identical. And Colossal has entered into a partnership with the conservation organization Re:wild to use biotechnology in its projects.


#### Abstractive Summarization


```python
from transformers import pipeline
```


```python
# Using google/pegasus-xsum for summarization

summarizer = pipeline("summarization", model="google/pegasus-xsum", framework="pt")

summary = summarizer(article, max_length=200, min_length=50, do_sample=False)

print(summary[0]['summary_text'])
```

    Some weights of PegasusForConditionalGeneration were not initialized from the model checkpoint at google/pegasus-xsum and are newly initialized: ['model.decoder.embed_positions.weight', 'model.encoder.embed_positions.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Device set to use cpu
    Token indices sequence length is longer than the specified maximum sequence length for this model (1928 > 512). Running this sequence through the model will result in indexing errors



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[14], line 5
          1 # Using google/pegasus-xsum for summarization
          3 summarizer = pipeline("summarization", model="google/pegasus-xsum", framework="pt")
    ----> 5 summary = summarizer(article, max_length=200, min_length=50, do_sample=False)
          7 print(summary[0]['summary_text'])


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/pipelines/text2text_generation.py:280, in SummarizationPipeline.__call__(self, *args, **kwargs)
        256 def __call__(self, *args, **kwargs):
        257     r"""
        258     Summarize the text(s) given as inputs.
        259 
       (...)
        278           ids of the summary.
        279     """
    --> 280     return super().__call__(*args, **kwargs)


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/pipelines/text2text_generation.py:173, in Text2TextGenerationPipeline.__call__(self, *args, **kwargs)
        144 def __call__(self, *args, **kwargs):
        145     r"""
        146     Generate the output text(s) using text(s) given as inputs.
        147 
       (...)
        170           ids of the generated text.
        171     """
    --> 173     result = super().__call__(*args, **kwargs)
        174     if (
        175         isinstance(args[0], list)
        176         and all(isinstance(el, str) for el in args[0])
        177         and all(len(res) == 1 for res in result)
        178     ):
        179         return [res[0] for res in result]


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/pipelines/base.py:1362, in Pipeline.__call__(self, inputs, num_workers, batch_size, *args, **kwargs)
       1354     return next(
       1355         iter(
       1356             self.get_iterator(
       (...)
       1359         )
       1360     )
       1361 else:
    -> 1362     return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/pipelines/base.py:1369, in Pipeline.run_single(self, inputs, preprocess_params, forward_params, postprocess_params)
       1367 def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
       1368     model_inputs = self.preprocess(inputs, **preprocess_params)
    -> 1369     model_outputs = self.forward(model_inputs, **forward_params)
       1370     outputs = self.postprocess(model_outputs, **postprocess_params)
       1371     return outputs


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/pipelines/base.py:1269, in Pipeline.forward(self, model_inputs, **forward_params)
       1267     with inference_context():
       1268         model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
    -> 1269         model_outputs = self._forward(model_inputs, **forward_params)
       1270         model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
       1271 else:


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/pipelines/text2text_generation.py:202, in Text2TextGenerationPipeline._forward(self, model_inputs, **generate_kwargs)
        199 if "generation_config" not in generate_kwargs:
        200     generate_kwargs["generation_config"] = self.generation_config
    --> 202 output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        203 out_b = output_ids.shape[0]
        204 if self.framework == "pt":


    File /Applications/anaconda3/lib/python3.12/site-packages/torch/utils/_contextlib.py:116, in context_decorator.<locals>.decorate_context(*args, **kwargs)
        113 @functools.wraps(func)
        114 def decorate_context(*args, **kwargs):
        115     with ctx_factory():
    --> 116         return func(*args, **kwargs)


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/generation/utils.py:2067, in GenerationMixin.generate(self, inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, negative_prompt_ids, negative_prompt_attention_mask, **kwargs)
       2063         raise ValueError("`attention_mask` passed to `generate` must be 2D.")
       2065 if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
       2066     # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
    -> 2067     model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
       2068         inputs_tensor, model_kwargs, model_input_name, generation_config
       2069     )
       2071 # 5. Prepare `input_ids` which will be used for auto-regressive generation
       2072 if self.config.is_encoder_decoder:


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/generation/utils.py:652, in GenerationMixin._prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs, model_input_name, generation_config)
        650 encoder_kwargs["return_dict"] = True
        651 encoder_kwargs[model_input_name] = inputs_tensor
    --> 652 model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)  # type: ignore
        654 return model_kwargs


    File /Applications/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
       1734     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1735 else:
    -> 1736     return self._call_impl(*args, **kwargs)


    File /Applications/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1747, in Module._call_impl(self, *args, **kwargs)
       1742 # If we don't have any hooks, we want to skip the rest of the logic in
       1743 # this function, and just call forward.
       1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1745         or _global_backward_pre_hooks or _global_backward_hooks
       1746         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1747     return forward_call(*args, **kwargs)
       1749 result = None
       1750 called_always_called_hooks = set()


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:743, in PegasusEncoder.forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
        740 if inputs_embeds is None:
        741     inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    --> 743 embed_pos = self.embed_positions(input_shape)
        745 hidden_states = inputs_embeds + embed_pos
        747 hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)


    File /Applications/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
       1734     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1735 else:
    -> 1736     return self._call_impl(*args, **kwargs)


    File /Applications/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1747, in Module._call_impl(self, *args, **kwargs)
       1742 # If we don't have any hooks, we want to skip the rest of the logic in
       1743 # this function, and just call forward.
       1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1745         or _global_backward_pre_hooks or _global_backward_hooks
       1746         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1747     return forward_call(*args, **kwargs)
       1749 result = None
       1750 called_always_called_hooks = set()


    File /Applications/anaconda3/lib/python3.12/site-packages/torch/utils/_contextlib.py:116, in context_decorator.<locals>.decorate_context(*args, **kwargs)
        113 @functools.wraps(func)
        114 def decorate_context(*args, **kwargs):
        115     with ctx_factory():
    --> 116         return func(*args, **kwargs)


    File /Applications/anaconda3/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:103, in PegasusSinusoidalPositionalEmbedding.forward(self, input_ids_shape, past_key_values_length)
         99 bsz, seq_len = input_ids_shape[:2]
        100 positions = torch.arange(
        101     past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        102 )
    --> 103 return super().forward(positions)


    File /Applications/anaconda3/lib/python3.12/site-packages/torch/nn/modules/sparse.py:190, in Embedding.forward(self, input)
        189 def forward(self, input: Tensor) -> Tensor:
    --> 190     return F.embedding(
        191         input,
        192         self.weight,
        193         self.padding_idx,
        194         self.max_norm,
        195         self.norm_type,
        196         self.scale_grad_by_freq,
        197         self.sparse,
        198     )


    File /Applications/anaconda3/lib/python3.12/site-packages/torch/nn/functional.py:2551, in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
       2545     # Note [embedding_renorm set_grad_enabled]
       2546     # XXX: equivalent to
       2547     # with torch.no_grad():
       2548     #   torch.embedding_renorm_
       2549     # remove once script supports set_grad_enabled
       2550     _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    -> 2551 return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


    IndexError: index out of range in self

