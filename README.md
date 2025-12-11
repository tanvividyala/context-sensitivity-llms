# High-Context vs. Low-Context Sensitivity in LLMs and ‘WEIRD’ Cultural Bias in Training Corpora
**Tanvi Vidyala** <br>
*COGS 150 Fall 2025 Final Project*

## Research Question
Do Large Language Models assign significantly lower surprisal to low-context speech acts compared to high-context equivalents? Additionally, do multilingual models perform better at high-context tasks?

## Background
In 1976, anthropologist Edward T. Hall proposed a fundamental distinction in human interaction and communication. Through his work in Beyond Culture he examined the divide between high-context societies, in which meaning is deeply embedded in relationships and the environment, and low-context cultures, where information is vested almost entirely in the explicit code of language.

Meaning in high-context societies is often “internalized in the person” or embedded in the physical situation, tending to rely on implicit cues and deep social bonds. Sensitivity towards high-context environments requires an instinctive understanding of cultural cues and shared knowledge that may appear ambiguous to individuals outside of these groups. Many Asian and African nations are classified under this high-context label. On the other hand, low-context cultures such as the United States, Canada, and many Western European nations, prioritize direct and explicit verbal communication. The mass of information transferred in these cultures is explicitly coded rather than being context-dependent. Hall’s framework is linked to linguistics, through the lens of explicit and implicit speech acts. Although utterances are typically categorized by function, the manner in which the acts are delivered is socially conditioned as either explicit (low-context) or implicit (high-context).

These same patterns find their way into the vast corpuses of data that Large Language Models (LLMs) are trained on. A study by Tao et al. conducted in 2024 confirms that the most popular LLMs are trained on corpora that overrepresent certain parts of the world and exhibit a bias favoring Western cultural values. These corpora are dominated by English text produced by internet users in WEIRD (Western, Educated, Industrialized, Rich, and Democratic) nations. This data collection leaves out the unique nuances in the lexicon of nearly 2.9 Billion people who do not have internet access, many of whom are from high-context linguistic groups. Data derived from WEIRD samples tend to represent an analytical perspective on stimuli, focusing largely on an object’s attributes and categories rather than relationships or cultural context . 

## Hypothesis
I hypothesize that LLMs will show lower surprisal for low-context stimuli because their training data draws heavily from WEIRD settings that favor direct and explicit communication. I believe this is the case because models that are trained on corpora dominated by English text will most likely inherit the structural patterns and pragmatic expectations of these environments, treating explicit language as the default form of conveying intent. High-context phrasing is reliant on complex background knowledge and indirect cues that appear less often in LLM training datasets, so models should treat these forms as less predictable.

## Methods
### Model Selection
To test this idea across different training setups, I used four LLMs from the Hugging Face transformers library. I selected two causal models, `distilgpt2` and `gpt2-medium`. Both of these models operate with a next-token prediction objective and represent a common family of English-dominant architectures. I additionally added two masked models, `mBERT` and `XLM-R`, which are known to allow comparison across multilingual training corpora. The mask-prediction objective and broader language coverage can create a useful contrast with the causal models.

## Stimuli Construction
I designed ten minimally paired stimuli, producing twenty items in total. Each pair consisted of an English sentence that represented a different type of speech act such as a request, refusal, disagreement, or correction. Every pair had a low-context version that expressed meaning directly and a high-context version that relied on subtle cues or softer phrasing. This format matched the idea that high-context environments embed meaning in the situation, while low-context environments prefer explicit coding of intent.

| Condition   | Prefix                                              | Critical Word | Context            |
|-------------|------------------------------------------------------|----------------|---------------------|
| LowContext  | Please close the window                              | now            | Request             |
| HighContext | It is getting a bit                                  | chilly         | Request             |
| LowContext  | Your report has several                              | errors         | Negative Feedback   |
| HighContext | Your report might benefit from a                     | review         | Negative Feedback   |
| LowContext  | I cannot attend your dinner party on                 | Friday         | Refusal             |
| HighContext | Friday might be a little                             | difficult      | Refusal             |
| LowContext  | I disagree with your proposal because it is too      | expensive      | Disagreement        |
| HighContext | We might want to explore more budget-friendly        | options        | Disagreement        |
| LowContext  | Send me the final files by                           | 5 PM           | Deadline            |
| HighContext | It would be great to receive the files before the end of the | day     | Deadline            |
| LowContext  | You sent the email to the wrong                      | address        | Correction          |
| HighContext | It seems the email may have gone to a different      | address        | Correction          |
| LowContext  | Stop talking so                                      | loudly         | Request             |
| HighContext | I’m finding it a little hard to                      | concentrate    | Request             |
| LowContext  | Take this food back because it is                    | cold           | Complaint           |
| HighContext | This dish is not quite as warm as I                  | expected       | Complaint           |
| LowContext  | I am                                                 | leaving        | Departure           |
| HighContext | I should probably get                                | going          | Departure           |
| LowContext  | Lend me your                                         | pen            | Request             |
| HighContext | Do you happen to have a spare                        | pen            | Request             |

*Table 1. Stimuli for High and Low Context Sentences*

### Assessing Surprisal Through Functions
Surprisal was calculated as a proxy for the model's "surprise" to see how unexpected the missing word is. I defined two distinct functions to accommodate the architectural differences between the models:

`casual_surprisal()`: For `distilgpt2` and `gpt2-medium`, I calculated surprisal of the critical word that followed the prefix. These models predict the next token in sequence, so this method is aligned with their training objective. <br>
`masked_surprisal()`: For `mBERT` and `XLM-R`, I placed a `[MASK]` token at the end of the prefix. I extracted the logits for the mask position to compute the probability of the critical word. If the tokenizer split the word into several sub-tokens, I used the mean surprisal across those pieces.

### Procedure
I ran each of the twenty stimuli through all four models. This process produced 80 total data points. For each case, I recorded the model, the condition, the critical word, and the surprisal value in a pandas dataframe.

| model        | condition   | critical     | surprisal   |
|--------------|-------------|--------------|-------------|
| distilgpt2   | LowContext  | cold         | 5.225095    |
| gpt2-medium  | LowContext  | cold         | 4.206719    |
| mBERT        | LowContext  | cold         | 3.946653    |
| XLM-R        | LowContext  | cold         | 2.817520    |
| distilgpt2   | HighContext | chilly       | 9.574214    |
| ....         | ....         | ....          | ....         |
| XLM-R        | LowContext  | temperature  | 4.019905    |
| distilgpt2   | HighContext | room         | 2.339399    |
| gpt2-medium  | HighContext | room         | 2.010232    |
| mBERT        | HighContext | room         | 4.086168    |
| XLM-R        | HighContext | room         | 5.441202    |

*Table 2. Final Results of Test with Surprisal Rates*

## Results
Consistent with my hypothesis, all four models exhibited lower mean surprisal for Low-Context stimuli compared to High-Context stimuli. This indicates that the models found direct, explicit communication norms (typical of WEIRD cultures) significantly more predictable than indirect, high-context alternatives.

<iframe src="assets/bar_plot.html" width="1200" height="600" frameborder="0" ></iframe>
*Figure 1. Mean Surprisal Across Models (High vs. Low Context)*

As illustrated in Figure 1, the disparity was observable across both causal and masked architectures. Causal models, like distilgpt2 and gpt2-medium, showed a distinct preference for low-context completions. For example, distilgpt2 assigned a surprisal of approximately 5.23 bits to the low-context critical word "cold" (in a food complaint scenario), compared to 9.57 bits for the high-context equivalent "chilly". Additionally, the effect held and was very pronounced in the masked multilingual models. XLM-R displayed the largest divergence, with mean surprisal values for high-context stimuli reaching significantly higher levels than their low-context counterparts.

### Distributional Analysis
<iframe src="assets/violin_plot.html" width="1200" height="600" frameborder="0" ></iframe>
*Figure 2. Per-Model Surprisal Differences Violin Plot (High vs. Low Context)*

<iframe src="assets/boxplot.html" width="1200" height="600" frameborder="0" ></iframe>
*Figure 3. Per-Model Surprisal Differences Box Plot (High vs. Low Context)*

Violin (Fig 2) and box plot (Fig 3) analyses of the results further confirm that the distribution of surprisal values for High-Context items (red) was consistently higher and more variable than for Low-Context items (blue). While mBERT showed a slightly tighter distribution, the median surprisal for High-Context stimuli remained strictly higher than for Low-Context stimuli across every model tested. These findings support the conclusion that LLMs possess a structural bias toward the explicit communication styles prevalent in their predominantly Western training data.

# Discussion
Across all four models, low-context formulations received lower surprisal than high-context equivalents, suggesting direct, explicit realizations are encoded as more expected continuations. This aligns with the idea that LLMs, trained primarily on WEIRD-dominated and low-context-leaning corpora, internalize explicitness as a default communicative norm. At the same time, the English-only, decontextualized design and genre biases of the training data mean that the effect cannot be straightforwardly attributed to “culture” alone.

This study faces several potential confounding variables. All stimuli were written in English and modeled on relatively idealized examples of indirectness, which may reflect English-specific genre and register conventions rather than universal high-context norms. Differences in token frequency and subword segmentation between critical words (e.g., “now” vs. “chilly”) could also influence surprisal independently of contextual explicitness. Model-level factors can also further complicate the interpretability of this study. Each of the four models used differ in training objectives, tokenization schemes, and (for the multilingual models) undocumented mixtures of non-English data. With only 10 minimal pairs and a narrow set of speech acts, the experiment may also overfit to details of this small dataset rather than capturing a generalizable effect.

Even with these constraints, the study highlights a meaningful concern in the accessibility of LLMs. These systems operate on a global scale, yet they may favor direct speech patterns associated with WEIRD cultures. High-context phrasing may appear marked or unexpected to the model. This creates pressure toward explicit forms of communication that do not match the needs of users from high-context backgrounds who may perceive low-context word predictions as impolite. Future work towards this question should involve larger stimulus sets, more naturalistic examples, and languages beyond English. Studies that juxtapose model surprisal with human judgments across cultural groups can help clarify how LLMs learn and reproduce context-based communication norms, making these models helpful for people worldwide.

## Works Cited
* Anadolu Agency. *["2.9B people still offline, while 4.9B used internet in 2021: UN"](https://www.aa.com.tr/en/world/29b-people-still-offline-while-49b-used-internet-in-2021-un/2435060).* Anadolu Agency, 30 Nov. 2021.

* Hall, Edward T., and Mildred Reed Hall. *["Key Concepts: Underlying Structures of Culture"](https://www.csun.edu/~sm60012/Intercultural/Key%20Concepts%20-%20Hall%20and%20Hall%20-%201.pdf).* Understanding Cultural Differences, Intercultural Press, 1990, pp. 199–202.

* Resnik, Philip. *["Large Language Models are Biased Because They Are Large Language Models"](https://arxiv.org/abs/2406.13138).* arXiv, 2024.

* Tao, Yan, et al. *["Cultural Bias and Cultural Alignment of Large Language Models"](https://arxiv.org/abs/2311.14096).* arXiv, 2024.

* Trott, Sean. *["GPT-4 Is ‘WEIRD’ — What Should We Do About It?"](https://seantrott.substack.com/p/gpt-4-is-weirdwhat-should-we-do-about).* The Counterfactual, 12 Feb. 2024.

* Sheposh, Richard. *["High-context and low-context cultures"](https://www.ebsco.com/research-starters/communication-and-mass-media/high-context-and-low-context-cultures).* Research Starters, EBSCO, 2025.

* Zaib, Zaha, Syed Kazim Shah, and Muhammad Ilyas Mahmood. *["A Comparative Analysis of Searle’s Speech Act Theory and Cohen’s Model: An Exploration of the Social Contexts in Which Explicit and Implicit Speech Acts Are Used"](https://ojs.jdss.org.pk/journal/article/view/1093).* Journal of Development and Social Sciences, vol. 4, no. 3, 30 Sept. 2023, pp. 1223–1236.

