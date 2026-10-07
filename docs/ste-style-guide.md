# ASD-STE100 Simplified Technical English: the standard for this repository

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. Rules for the text

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, "test" is a noun or a verb, "check" is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: "prepare", "do", "find", "get", "make".
4. Do not use an "-ing" form as a noun or an adjective ("the running job", "after indexing").
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and "can" for a possibility.
8. Keep the articles "a", "an" and "the" in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: "If the index is stale, build it again."
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase ("The cost model") or an imperative ("Run the demo").
   Do not start a heading with an "-ing" form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or "check that" |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the fridge2fork documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **recipe file** | The CSV, JSON Lines or Parquet file with the recipes (`FRIDGE2FORK_RECIPES`). | dataset (for the file), corpus |
| **recipe** | One row of the recipe file with a name, ingredients and optional steps. | dish, document, item |
| **raw ingredient** | An ingredient string as the recipe file gives it, for example `2 large tomatoes, chopped`. | ingredient line, entry |
| **canonical name** | The normalised name of an ingredient, for example `tomato`. | clean name, token, lemma |
| **vocabulary** | All canonical names in the recipe data. | dictionary, word list |
| **user ingredients** | The ingredients that the user types. | query items, pantry (for the user list) |
| **pantry item** | An ingredient that the score treats as owned: `salt`, `pepper`, `water`, `vegetable oil`, `oil`, `olive oil`, `sugar`. | staple, basic |
| **unknown ingredient** | A user ingredient with no canonical name in the vocabulary and no close spelling. | OOV word, missing word |
| **correction** | The change of a user ingredient to a close vocabulary name. | fix, fuzzy match (in prose) |
| **embedder** | The component that gives a vector for each canonical name: PPMI + SVD or Word2Vec. | encoder, model (for the embedder) |
| **ingredient vector** | The unit-length vector of one canonical name. | embedding (as a noun), word vector |
| **recipe vector** | The unit-length mean of the ingredient vectors of a recipe. | recipe embedding, document vector |
| **query vector** | The recipe vector of the user ingredients. | user embedding |
| **index** | The saved files of `fridge2fork build`: recipes, ingredient vectors, vocabulary, manifest. | model file, database |
| **vector index** | The cosine search structure over recipe vectors (`numpy` or `faiss`). | FAISS index (for both), vector store |
| **inverted index** | The recipe-ingredient incidence matrix for overlap counts. | lookup table, posting list |
| **candidate** | A recipe that a retriever returns before the score. | hit, result |
| **dense retriever** | The vector index search with the query vector. | semantic search, embedding search |
| **overlap retriever** | The inverted index search by shared ingredients. | keyword search, lexical search |
| **coverage** | Share of the recipe ingredients that the user has, pantry items included. | match rate, completeness |
| **use** | Share of the user ingredients that the recipe uses. | relevance, utilisation |
| **missing ingredient** | A recipe ingredient that the user does not have and that is not a pantry item. | absent item, lacking item |
| **score** | The weighted sum of coverage, use, cosine and the missing penalty. | final score, rank score |
| **allergen group** | A named set of keywords, for example `dairy`. | allergy class, category |
| **explanation** | The text that tells why a recipe fits, with substitutions and the real steps. | description, LLM answer |
| **chat model** | The optional model that writes the "why it fits" part. | LLM (in prose), GPT, bot |
| **reply check** | The rule set that accepts or refuses a chat model reply. | guard, validator |
| **variant** | One ranking setup in the evaluation: `dense`, `overlap` or `hybrid`. | mode, configuration |
| **held-out recipe** | A recipe that the evaluation hides from the embedder fit and searches for. | test recipe, target |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **normalise** | Change a raw ingredient into a canonical name. |
| **correct** | Change a user ingredient to the closest vocabulary name. |
| **build** | Fit the embedder, make the vector index and the inverted index, and save them. |
| **load** | Read a saved index. |
| **retrieve** | Get candidates from the dense retriever and the overlap retriever. |
| **score** | Calculate the score of one candidate for one query. |
| **filter** | Remove candidates that fail the maximum-missing, allergen or vegetarian rule. |
| **explain** | Make the explanation of one recommendation. |
| **refuse** | Reject a chat model reply and use the template explanation. |
| **evaluate** | Hide ingredients of held-out recipes and measure recall@k and MRR. |
