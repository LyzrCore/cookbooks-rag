"""
200-query bank for the multi-KB oneshot_rag vs agentic_rag routing benchmark.

Each query carries:
  - category    : query shape (factoid, sibling, cross_kb, oos, chitchat, ...)
  - question    : the user message
  - expected_kb : set of canonical KB base-names the planner SHOULD route to.
                  * single-target queries  -> one name
                  * cross_kb queries       -> 2+ names (any/all scored, see harness)
                  * sibling queries        -> exactly one of a close pair (precision test)
                  * oos                    -> empty set (planner ideally retrieves nothing useful;
                                              a graceful "not covered" is the correct outcome)
                  * chitchat               -> empty set AND needs_retrieval should be False

KB base-names (must match setup_dev_multi_kb.py KBSpec.name):
  kb_transformers_foundation_models  kb_llm_evaluation
  kb_dem_superresolution             kb_authors_dem_project
  kb_ml_textbooks                    kb_nlp
  kb_python_programming              kb_webrtc_networking
  kb_internship_report_templates     kb_wire_commercial   kb_wire_consumer
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MKQuery:
    category: str
    question: str
    expected_kb: frozenset = field(default_factory=frozenset)


def _q(cat, question, *kbs):
    return MKQuery(cat, question, frozenset(kbs))


TRANSFORMERS = "kb_transformers_foundation_models"
LLM_EVAL = "kb_llm_evaluation"
DEM = "kb_dem_superresolution"
AUTHOR = "kb_authors_dem_project"
ML = "kb_ml_textbooks"
NLP = "kb_nlp"
PY = "kb_python_programming"
WEBRTC = "kb_webrtc_networking"
INTERN = "kb_internship_report_templates"
WIRE_C = "kb_wire_commercial"
WIRE_R = "kb_wire_consumer"


QUERIES: list[MKQuery] = [
    # ===================== KB1 transformers (13) =====================
    _q("factoid", "How does the self-attention mechanism work in a transformer?", TRANSFORMERS),
    _q("factoid", "What is multi-head attention and why is it used?", TRANSFORMERS),
    _q("factoid", "What is the role of positional encoding in transformers?", TRANSFORMERS),
    _q("factoid", "What problem does 'Attention Is All You Need' solve over RNNs?", TRANSFORMERS),
    _q("factoid", "What are Kolmogorov-Arnold Networks (KAN)?", TRANSFORMERS),
    _q("factoid", "How do KANs differ from multi-layer perceptrons?", TRANSFORMERS),
    _q("factoid", "What is the parameter count of Nemotron-4 340B?", TRANSFORMERS),
    _q("factoid", "What training data was used for Nemotron-4 340B?", TRANSFORMERS),
    _q("enumeration", "List the components of the transformer encoder block.", TRANSFORMERS),
    _q("specific", "What is the dimensionality of the model in the original Transformer paper?", TRANSFORMERS),
    _q("factoid", "How is scaled dot-product attention computed?", TRANSFORMERS),
    _q("factoid", "What activation functions do KANs place on edges?", TRANSFORMERS),
    _q("factoid", "Does Nemotron-4 340B include a reward model?", TRANSFORMERS),

    # ===================== KB2 llm_evaluation (12) =====================
    _q("factoid", "How does the Lynx hallucination evaluation model work?", LLM_EVAL),
    _q("factoid", "What is Lynx used to detect?", LLM_EVAL),
    _q("factoid", "What benchmark does Lynx use for faithfulness evaluation?", LLM_EVAL),
    _q("factoid", "Is Lynx open source?", LLM_EVAL),
    _q("factoid", "How does Lynx compare to GPT-4 as a hallucination judge?", LLM_EVAL),
    _q("enumeration", "What categories of hallucination does the Lynx paper define?", LLM_EVAL),
    _q("factoid", "What model size is the Lynx evaluator?", LLM_EVAL),
    _q("factoid", "How is faithfulness scored in the Lynx framework?", LLM_EVAL),
    _q("factoid", "What is HaluBench?", LLM_EVAL),
    _q("specific", "What accuracy does Lynx report on hallucination detection?", LLM_EVAL),
    _q("factoid", "Why use a fine-tuned model instead of a prompt for hallucination eval?", LLM_EVAL),
    _q("factoid", "What domains does the Lynx evaluation cover?", LLM_EVAL),

    # ===================== KB3 dem_superresolution (14) =====================
    _q("factoid", "How does D-SRCAGAN super-resolve digital elevation models?", DEM),
    _q("factoid", "What GAN architectures are used for DEM super-resolution?", DEM),
    _q("factoid", "How is sparse representation used for lunar DEM reconstruction?", DEM),
    _q("factoid", "What is a local implicit function representation for DEMs?", DEM),
    _q("factoid", "How does deep learning enhance digital elevation models?", DEM),
    _q("factoid", "What loss functions are used in DEM SRGAN?", DEM),
    _q("factoid", "How is pixel-scale lunar DEM super-resolution achieved with GANs?", DEM),
    _q("enumeration", "List the evaluation metrics used for DEM super-resolution quality.", DEM),
    _q("factoid", "What is the upscaling factor in the DEM super-resolution papers?", DEM),
    _q("factoid", "How do the DEM papers handle terrain edge preservation?", DEM),
    _q("factoid", "What datasets are used to train DEM super-resolution models?", DEM),
    _q("factoid", "What is the role of the discriminator in DEM SRGAN?", DEM),
    _q("specific", "What PSNR improvement does D-SRCAGAN report?", DEM),
    _q("factoid", "How does conditional GAN conditioning work for DEM super-resolution?", DEM),

    # ===================== KB4 authors_dem_project (12) =====================
    _q("factoid", "What was Parshva's DEM enhancement project about?", AUTHOR),
    _q("factoid", "What methodology did the author use to enhance DEMs from high-resolution imagery?", AUTHOR),
    _q("factoid", "What were the results of the 8th semester DEM project?", AUTHOR),
    _q("factoid", "What organization was the author's DEM project done at?", AUTHOR),
    _q("specific", "What is the title of the author's DEM enhancement report?", AUTHOR),
    _q("factoid", "Did the author receive a project completion certificate?", AUTHOR),
    _q("factoid", "What input data did the author use for DEM enhancement?", AUTHOR),
    _q("factoid", "What tools or frameworks did the author use in the DEM project?", AUTHOR),
    _q("factoid", "What problem statement did the author's final report address?", AUTHOR),
    _q("enumeration", "What chapters are in the author's 8th semester final report?", AUTHOR),
    _q("factoid", "What conclusions did the author draw in the DEM project?", AUTHOR),
    _q("factoid", "Who certified the author's project?", AUTHOR),

    # ===================== KB5 ml_textbooks (14) =====================
    _q("factoid", "How do you design a production machine learning system?", ML),
    _q("factoid", "What is a learning curve in machine learning?", ML),
    _q("factoid", "What are best practices for deep learning model training?", ML),
    _q("factoid", "How does feature engineering improve model performance?", ML),
    _q("factoid", "What is the bias-variance tradeoff?", ML),
    _q("factoid", "How do you handle data distribution shift in production ML?", ML),
    _q("factoid", "What is transfer learning in deep learning?", ML),
    _q("factoid", "How does batch normalization help training?", ML),
    _q("enumeration", "List the stages of an iterative ML system lifecycle.", ML),
    _q("factoid", "What is the difference between batch and online prediction?", ML),
    _q("factoid", "How do you choose a loss function for a classification model?", ML),
    _q("factoid", "What is model monitoring and why does it matter in production?", ML),
    _q("factoid", "How do convolutional neural networks process images?", ML),
    _q("factoid", "What is regularization and how does dropout work?", ML),

    # ===================== KB6 nlp (12) =====================
    _q("factoid", "What is tokenization in natural language processing?", NLP),
    _q("factoid", "How does text classification work?", NLP),
    _q("factoid", "What is a language model in NLP?", NLP),
    _q("factoid", "What is named entity recognition?", NLP),
    _q("factoid", "How does part-of-speech tagging work?", NLP),
    _q("factoid", "What are word embeddings?", NLP),
    _q("factoid", "What is stemming versus lemmatization?", NLP),
    _q("factoid", "How is sentiment analysis performed?", NLP),
    _q("enumeration", "List common NLP preprocessing steps.", NLP),
    _q("factoid", "What is the bag-of-words model?", NLP),
    _q("factoid", "How does TF-IDF weighting work?", NLP),
    _q("factoid", "What is parsing in natural language processing?", NLP),

    # ===================== KB7 python_programming (14) =====================
    _q("factoid", "How do you scrape a website with Python?", PY),
    _q("factoid", "What is a Python list comprehension?", PY),
    _q("factoid", "How do you call the Gemini API from Python?", PY),
    _q("factoid", "How do you handle HTTP requests in Python web scraping?", PY),
    _q("factoid", "What is the difference between a Python list and a tuple?", PY),
    _q("factoid", "How do Python decorators work?", PY),
    _q("factoid", "How do you parse HTML with BeautifulSoup?", PY),
    _q("factoid", "What are Python generators and the yield keyword?", PY),
    _q("enumeration", "List Python data types covered in Learning Python.", PY),
    _q("factoid", "How do you handle exceptions in Python?", PY),
    _q("factoid", "How is Python used with Bard or Gemini for prompting?", PY),
    _q("factoid", "What is a Python virtual environment?", PY),
    _q("factoid", "How do you respect robots.txt when scraping with Python?", PY),
    _q("factoid", "How do Python dictionaries work internally?", PY),

    # ===================== KB8 webrtc_networking (12) =====================
    _q("factoid", "How does WebRTC establish a peer connection?", WEBRTC),
    _q("factoid", "What is ICE in WebRTC?", WEBRTC),
    _q("factoid", "What is the difference between STUN and TURN?", WEBRTC),
    _q("factoid", "What is signaling in WebRTC?", WEBRTC),
    _q("factoid", "How do WebRTC data channels work?", WEBRTC),
    _q("factoid", "What is SRTP in WebRTC?", WEBRTC),
    _q("factoid", "How does NAT traversal work in WebRTC?", WEBRTC),
    _q("factoid", "What is an SDP offer and answer?", WEBRTC),
    _q("enumeration", "List the steps to set up a WebRTC peer connection.", WEBRTC),
    _q("factoid", "How does WebRTC handle media encryption?", WEBRTC),
    _q("factoid", "What transport protocol does WebRTC use for media?", WEBRTC),
    _q("factoid", "Why is signaling not standardized in WebRTC?", WEBRTC),

    # ===================== KB9 internship_report_templates (12) =====================
    _q("factoid", "What sections must a college internship report include?", INTERN),
    _q("factoid", "What are the internship report formatting guidelines?", INTERN),
    _q("factoid", "How do you fill the student daily diary log?", INTERN),
    _q("factoid", "What goes on the internship report title page?", INTERN),
    _q("factoid", "What is required in the certificate and completion certificate section?", INTERN),
    _q("factoid", "How should the acknowledgement and abstract be structured?", INTERN),
    _q("enumeration", "List the chapters required in the internship report template.", INTERN),
    _q("factoid", "What does the weekly log format require?", INTERN),
    _q("factoid", "What feedback is expected from the industry expert?", INTERN),
    _q("specific", "What is the subject code on the internship report guidelines?", INTERN),
    _q("factoid", "What does a demo internship report look like?", INTERN),
    _q("factoid", "What is the page format for the internship report?", INTERN),

    # ===================== KB10 wire_commercial (14) =====================
    _q("factoid", "What documents are required to initiate a commercial wire transfer?", WIRE_C),
    _q("factoid", "Who needs to approve a commercial wire transfer request?", WIRE_C),
    _q("factoid", "What is in the wire transfer submission checklist?", WIRE_C),
    _q("factoid", "What does the Bravo Enterprises wire agreement specify?", WIRE_C),
    _q("factoid", "What is the Bath Planet of Chicago wire request for?", WIRE_C),
    _q("specific", "What is the wire amount on the Bath Planet of Chicago request dated 01/21/2026?", WIRE_C),
    _q("factoid", "Who is the originator on the Bravo Enterprises wire?", WIRE_C),
    _q("factoid", "What is the Midwest Imaging wire approval about?", WIRE_C),
    _q("enumeration", "List the fields a commercial wire request form must include.", WIRE_C),
    _q("factoid", "What happens if a commercial wire submission is incomplete?", WIRE_C),
    _q("factoid", "What are the terms of the manual wire transfer agreement?", WIRE_C),
    _q("specific", "What is the beneficiary on the Commercial Sample 2 wire?", WIRE_C),
    _q("factoid", "What approvals are needed for a high-value commercial wire?", WIRE_C),
    _q("factoid", "What is the One More Wire Out request for Commercial Sample 3?", WIRE_C),

    # ===================== KB11 wire_consumer (10) =====================
    _q("factoid", "What documents are needed for a consumer wire transfer?", WIRE_R),
    _q("factoid", "Who is Vickie Ann Rossi in the consumer wire documents?", WIRE_R),
    _q("factoid", "What is the consumer wire approval process?", WIRE_R),
    _q("specific", "What is the date on the Vickie Ann Rossi wire?", WIRE_R),
    _q("factoid", "What approval was needed for the consumer wire?", WIRE_R),
    _q("factoid", "What does the consumer wire request specify?", WIRE_R),
    _q("factoid", "What is the beneficiary on the consumer Sample 1 wire?", WIRE_R),
    _q("enumeration", "List the documents in the consumer wire transfer package.", WIRE_R),
    _q("factoid", "Is the consumer wire an individual or business transfer?", WIRE_R),
    _q("factoid", "What amount is the consumer wire transfer for?", WIRE_R),

    # ===================== sibling-disambiguation (22) =====================
    # commercial vs consumer wire — must pick exactly the right one
    _q("sibling", "Is Vickie Ann Rossi a commercial or consumer customer, and what was her wire?", WIRE_R),
    _q("sibling", "Show me the commercial wire for Bravo Enterprises, not any consumer wire.", WIRE_C),
    _q("sibling", "What consumer (not commercial) wire approvals are on file?", WIRE_R),
    _q("sibling", "Among commercial wires only, which involve Bath Planet of Chicago?", WIRE_C),
    _q("sibling", "Which individual-person wire transfer is documented?", WIRE_R),
    _q("sibling", "Which business-entity wire agreements are documented?", WIRE_C),
    # transformers vs llm_eval — both LLM papers
    _q("sibling", "Which paper is about detecting hallucinations, not about attention?", LLM_EVAL),
    _q("sibling", "Which paper introduces the attention mechanism, not hallucination scoring?", TRANSFORMERS),
    _q("sibling", "For evaluating whether an LLM made something up, which resource applies?", LLM_EVAL),
    _q("sibling", "For the architecture behind GPT-style models, which resource applies?", TRANSFORMERS),
    # dem_superres (research) vs authors_project (own work)
    _q("sibling", "What did the AUTHOR personally do for DEM enhancement (not the general research)?", AUTHOR),
    _q("sibling", "What do the published research papers say about DEM super-resolution (not the author's own project)?", DEM),
    _q("sibling", "Which resource is Parshva's own certified project, not a third-party paper?", AUTHOR),
    _q("sibling", "Which resources are external GAN super-resolution research rather than the author's report?", DEM),
    # ml_textbooks vs nlp vs python
    _q("sibling", "For NLP-specific tokenization (not general ML or Python), where do I look?", NLP),
    _q("sibling", "For Python language syntax (not ML theory), where do I look?", PY),
    _q("sibling", "For production ML system design (not NLP, not Python syntax), where do I look?", ML),
    _q("sibling", "Where is web scraping covered — the Python books or the ML textbooks?", PY),
    _q("sibling", "Is language modeling covered in the NLP book or the Python books?", NLP),
    _q("sibling", "For deep learning fundamentals (not the NLP-only book), which resource?", ML),
    _q("sibling", "Which resource covers WebRTC, as opposed to web scraping?", WEBRTC),
    _q("sibling", "Which resource covers web scraping, as opposed to WebRTC real-time comms?", PY),

    # ===================== cross-KB fan-out (20) =====================
    _q("cross_kb", "Compare the attention mechanism in transformers with the GAN approach used in DEM super-resolution.", TRANSFORMERS, DEM),
    _q("cross_kb", "How does hallucination evaluation (Lynx) relate to general model evaluation in the ML textbooks?", LLM_EVAL, ML),
    _q("cross_kb", "Contrast KANs with the neural networks described in the deep learning textbook.", TRANSFORMERS, ML),
    _q("cross_kb", "How do the DEM super-resolution research papers compare to the author's own DEM project?", DEM, AUTHOR),
    _q("cross_kb", "Compare commercial and consumer wire transfer approval requirements.", WIRE_C, WIRE_R),
    _q("cross_kb", "How is Python used for NLP tasks described in the NLP book?", PY, NLP),
    _q("cross_kb", "Relate word embeddings (NLP) to feature representation in the ML textbooks.", NLP, ML),
    _q("cross_kb", "Compare how transformers and the NLP book each handle sequence modeling.", TRANSFORMERS, NLP),
    _q("cross_kb", "How might WebRTC data channels be implemented in Python?", WEBRTC, PY),
    _q("cross_kb", "Compare the GAN discriminator in DEM super-resolution with GANs in the deep learning textbook.", DEM, ML),
    _q("cross_kb", "What internship report sections would document a DEM enhancement project like the author's?", INTERN, AUTHOR),
    _q("cross_kb", "Compare the Nemotron-4 foundation model to the production-ML guidance in the textbooks.", TRANSFORMERS, ML),
    _q("cross_kb", "How would you evaluate hallucinations of an NLP language model using Lynx?", LLM_EVAL, NLP),
    _q("cross_kb", "Compare the wire transfer approval workflow to a college report approval workflow.", WIRE_C, INTERN),
    _q("cross_kb", "Relate transfer learning in the ML textbooks to fine-tuning the Lynx evaluator.", ML, LLM_EVAL),
    _q("cross_kb", "Compare lunar DEM reconstruction methods with the author's DEM enhancement approach.", DEM, AUTHOR),
    _q("cross_kb", "How do Python web-scraping tools relate to data collection for ML systems?", PY, ML),
    _q("cross_kb", "Compare scaled dot-product attention with attention as covered in the NLP material.", TRANSFORMERS, NLP),
    _q("cross_kb", "How do commercial wire checklists compare to internship report checklists?", WIRE_C, INTERN),
    _q("cross_kb", "Relate the Nemotron reward model to hallucination evaluation in Lynx.", TRANSFORMERS, LLM_EVAL),

    # ===================== out-of-corpus (10) =====================
    _q("oos", "What is the capital of France?"),
    _q("oos", "Who won the 2025 NBA Finals?"),
    _q("oos", "What's the weather forecast for tomorrow?"),
    _q("oos", "How do I bake a sourdough loaf?"),
    _q("oos", "What is the GDP of Japan in 2025?"),
    _q("oos", "Recommend a good Italian restaurant nearby."),
    _q("oos", "What time zone is Sydney in?"),
    _q("oos", "How many moons does Saturn have?"),
    _q("oos", "What is the stock price of Apple today?"),
    _q("oos", "Translate 'good morning' into Japanese."),

    # ===================== chitchat (needs_retrieval=False) (10) =====================
    _q("chitchat", "Hi, what can you help me with today?"),
    _q("chitchat", "Thanks, that was really helpful!"),
    _q("chitchat", "Good morning!"),
    _q("chitchat", "Who are you?"),
    _q("chitchat", "What topics do you know about?"),
    _q("chitchat", "Can you summarize what knowledge bases you have access to?"),
    _q("chitchat", "Great, appreciate it."),
    _q("chitchat", "Hello there."),
    _q("chitchat", "That makes sense, thank you."),
    _q("chitchat", "What else can you do?"),
]


def summary() -> dict:
    from collections import Counter
    c = Counter(q.category for q in QUERIES)
    return {"total": len(QUERIES), "by_category": dict(c)}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
