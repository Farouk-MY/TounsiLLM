# TounsiEval Evaluation Framework

## 1. Principles of Dialect Evaluation

Standard benchmark suites (MMLU, ARC, HellaSwag) do not assess colloquial North African dialects. Evaluating a Tunisian language model requires a benchmark reflecting authentic spoken and digital register realities.

**TounsiEval** establishes 14 distinct evaluation categories designed specifically for Tunisian Arabic:

1. **Tunisian Vocabulary**: Dialect lexical knowledge (`برشا`, `فيسع`, `بالشويا`, `كرومة`, `ملاوي`).
2. **Tunisian Grammar**: Negation syntax (`ما ... ش`), demonstratives (`هكّا`, `هاذي`), and Derja verb conjugations.
3. **Tunisian Comprehension**: Colloquial expressions, idiomatic blessings, and proverbs (`يعيشك ويرحم والديك`).
4. **Tunisian Generation**: Natural dialogue responses adhering to cultural pragmatics.
5. **Arabizi Comprehension**: Parsing Latin chat orthography with numeric glyphs (`3`, `5`, `7`, `8`, `9`).
6. **Arabizi Generation**: Synthesizing conversational Arabizi without unnatural formatting.
7. **Arabic-script Tunisian**: Pure Arabic-script Derja orthographic fluency.
8. **Tunisian/French Code-Switching**: Multi-lingual syntactic boundary management (`réunion`, `document`, `deploy`).
9. **Translation**: Derja $\leftrightarrow$ English, Derja $\leftrightarrow$ French, Derja $\leftrightarrow$ MSA.
10. **Everyday Conversation**: Greetings, pleasantries, customer-clerk interactions.
11. **Technical Tunisian**: Explaining computer science and AI concepts in Tunisian dialect (`API`, `base de données`).
12. **Reasoning in Tunisian**: Word math problems and logical reasoning articulated natively in Derja.
13. **Cultural & Contextual Understanding**: Knowledge of Tunisian cuisine, geography, history, and customs.
14. **Robustness to Spelling Variation**: Invariance across orthographic variants (`شنوة` vs `شنو` vs `chneya` vs `chnowa`).

---

## 2. Evaluation Metrics

### Automatic Language Modeling Metrics
- **Perplexity ($PPL$)**: Cross-entropy perplexity on held-out evaluation sets.
- **Token Efficiency**: Characters per token ($\text{Chars}/\text{Token}$) measuring subword compression.

### Human Evaluation Rubric (1–5 Likert Scale)
Because dialect fluency cannot be reduced to automated string matching, TounsiEval provides a human audit rubric:
- **Tunisian Naturalness (1–5)**: Authenticity to native speaker speech patterns.
- **Correctness (1–5)**: Semantic alignment with prompt instructions.
- **Meaning Preservation (1–5)**: Precision in maintaining intent across language boundaries.
- **Grammar (1–5)**: Conformity to Derja grammatical structures.
- **Dialect Adherence (1–5)**: Avoidance of unsolicited drift into Modern Standard Arabic or French.
- **Fluency (1–5)**: Flow, cadence, and absence of repetitive artifacts.
