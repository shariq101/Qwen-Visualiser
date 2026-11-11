"""
The probe corpus.

Every visualisation in this report is computed from forward passes over these
prompts (plus their continuations). They are deliberately chosen to separate
different kinds of computation, so that a layer's "function" has something to
be a function *of*:

  factual      multi-hop / single-hop recall, where the answer must be fetched
  syntax       subject-verb agreement with an attractor, where structure wins
  induction    repeated random tokens, where copying is the only strategy
  arithmetic   small integer arithmetic, where an algorithm must run
  code         python continuations, where indentation/syntax matter
  multiling    cross-lingual lookup
  negation     polarity, where the surface token is misleading
  idiom        non-compositional meaning

`target` is the token whose probability we track through the layers. It is the
greedy continuation of the base model, or the hand-specified answer.
"""

CORPUS = [
    # ---- factual recall -------------------------------------------------
    dict(id="fact-capital", kind="factual", group="factual",
         text="The capital of France is", target=" Paris"),
    dict(id="fact-eiffel", kind="factual", group="factual",
         text="The Eiffel Tower is located in the city of", target=" Paris"),
    dict(id="fact-author", kind="factual", group="factual",
         text="The author of Pride and Prejudice was", target=" Jane"),
    dict(id="fact-element", kind="factual", group="factual",
         text="The chemical symbol for gold is", target=" Au"),
    dict(id="fact-planet", kind="factual", group="factual",
         text="The largest planet in our solar system is", target=" Jupiter"),
    dict(id="fact-multihop", kind="factual", group="factual",
         text="The capital of the country whose flag is red with a white cross is",
         target=" Copenhagen"),

    # ---- syntax ---------------------------------------------------------
    dict(id="syn-agree-sg", kind="syntax", group="syntax",
         text="The key to the cabinets", target=" is"),
    dict(id="syn-agree-pl", kind="syntax", group="syntax",
         text="The keys to the cabinet", target=" are"),
    dict(id="syn-attract", kind="syntax", group="syntax",
         text="The key to the cabinets", target=" is"),
    dict(id="syn-attract2", kind="syntax", group="syntax",
         text="The report about the experiments", target=" was"),
    dict(id="syn-relclause", kind="syntax", group="syntax",
         text="The dog that chased the cats", target=" was"),
    dict(id="syn-prep", kind="syntax", group="syntax",
         text="She walked into the room and sat on the", target=" chair"),

    # ---- induction / copying -------------------------------------------
    dict(id="ind-1", kind="induction", group="induction",
         text="g7 k2 q9 g7 k2 q9 g7 k2", target=" q"),
    dict(id="ind-2", kind="induction", group="induction",
         text="apple river stone apple river stone apple", target=" river"),
    dict(id="ind-3", kind="induction", group="induction",
         text="z4 m8 t1 z4 m8 t1 z4 m8", target=" t"),
    dict(id="ind-names", kind="induction", group="induction",
         text="Mira talked to Jonas. Jonas talked to Mira. Mira talked to",
         target=" Jonas"),
    dict(id="ind-list", kind="induction", group="induction",
         text="alpha beta gamma delta alpha beta gamma", target=" delta"),

    # ---- arithmetic ------------------------------------------------------
    dict(id="arith-add", kind="arithmetic", group="arithmetic",
         text="3 + 4 =", target=" 7"),
    dict(id="arith-add2", kind="arithmetic", group="arithmetic",
         text="17 + 26 =", target=" 43"),
    dict(id="arith-sub", kind="arithmetic", group="arithmetic",
         text="52 - 17 =", target=" 35"),
    dict(id="arith-mul", kind="arithmetic", group="arithmetic",
         text="12 * 12 =", target=" 144"),
    dict(id="arith-carry", kind="arithmetic", group="arithmetic",
         text="99 + 1 =", target=" 100"),

    # ---- code ------------------------------------------------------------
    dict(id="code-def", kind="code", group="code",
         text="def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-",
         target="2)"),
    dict(id="code-loop", kind="code", group="code",
         text="for i in range(10):\n    print(i)\n\n# this loop prints the",
         target=" numbers"),
    dict(id="code-import", kind="code", group="code",
         text="import numpy as np\nx = np.array([1, 2, 3])\nprint(x.", target="shape"),
    dict(id="code-class", kind="code", group="code",
         text="class Dog:\n    def __init__(self, name):\n        self.name = name\n\ndog = Dog('Rex')\ndog.", target="name"),

    # ---- multilingual ----------------------------------------------------
    dict(id="ml-fr", kind="multilingual", group="multilingual",
         text="The French word for water is", target=" eau"),
    dict(id="ml-de", kind="multilingual", group="multilingual",
         text="The German word for house is", target=" Haus"),
    dict(id="ml-es", kind="multilingual", group="multilingual",
         text="The Spanish word for book is", target=" libro"),
    dict(id="ml-jp", kind="multilingual", group="multilingual",
         text="The Japanese word for cat is", target=" 猫"),

    # ---- negation / polarity --------------------------------------------
    dict(id="neg-1", kind="negation", group="negation",
         text="The movie was not good, it was", target=" bad"),
    dict(id="neg-2", kind="negation", group="negation",
         text="It is not cold outside, it is actually quite", target=" warm"),
    dict(id="neg-3", kind="negation", group="negation",
         text="The answer is not wrong, it is", target=" correct"),

    # ---- idiom / non-compositional --------------------------------------
    dict(id="idiom-1", kind="idiom", group="idiom",
         text="He kicked the bucket, which means he", target=" died"),
    dict(id="idiom-2", kind="idiom", group="idiom",
         text="It is raining cats and", target=" dogs"),
    dict(id="idiom-3", kind="idiom", group="idiom",
         text="Break a", target=" leg"),

    # ---- long-context / retrieval ---------------------------------------
    dict(id="long-name", kind="longctx", group="longctx",
         text=("The archive was catalogued in 1998 by a librarian named Teodora "
               "Vaskova, who worked alone for eleven years. The catalogue she "
               "built is still used today. The librarian who built the catalogue was"),
         target=" Teodora"),
    dict(id="long-copy", kind="longctx", group="longctx",
         text=("The password for the northern gate is FORTY-TWO-VIOLET. "
               "Guards rotate every six hours. The password for the northern gate is"),
         target=" FORTY"),
]


# Short, single-line prompts used for the causal ablation sweep. Kept short so
# the sweep (2 components x L layers forward passes) stays tractable on CPU.
ABLATION_IDS = [
    "fact-capital", "fact-element", "fact-author", "syn-agree-pl", "syn-attract",
    "ind-1", "ind-2", "arith-add", "arith-sub", "code-def", "ml-fr", "neg-1",
    "idiom-2", "long-name",
]

# Prompts used for the single-prompt deep dives (logit lens river, attribution
# graph, geometry trajectories).
DEEP_DIVE_IDS = ["fact-multihop", "syn-attract", "ind-1", "idiom-1"]
