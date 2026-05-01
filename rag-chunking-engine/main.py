"""
CLI entry point for the Smart Chunking & Retrieval System.

Usage examples
--------------
  # Pass a file path and query directly
  python main.py --doc my_document.txt --query "What is machine learning?"

  # Use the built-in demo
  python main.py --demo

  # Adjust chunking parameters
  python main.py --doc doc.pdf --query "climate change" --top-k 3 --overlap-window 8
"""

import argparse
import json
import sys
from chunking_system import analyse


# ──────────────────────────────────────────────────────────────────────────────
# Built-in demo document (used when --demo flag is passed)
# ──────────────────────────────────────────────────────────────────────────────

DEMO_DOCUMENT = """
# Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that enables systems to learn
from data and improve their performance without being explicitly programmed. It has
become one of the most transformative technologies of the 21st century.

## Supervised Learning

Supervised learning is the most widely used machine learning paradigm. In supervised
learning, the model is trained on labelled examples — input-output pairs — so that
it can predict outputs for unseen inputs. Common algorithms include linear regression,
decision trees, support vector machines, and neural networks.

Classification is a type of supervised learning where the output is a discrete label.
For instance, classifying emails as spam or not-spam. Regression, on the other hand,
predicts a continuous value such as house prices or stock returns.

## Unsupervised Learning

Unsupervised learning deals with unlabelled data. The goal is to discover hidden
structure or patterns. Clustering algorithms such as K-Means group similar data
points together. Dimensionality reduction techniques like PCA and t-SNE compress
high-dimensional data into lower-dimensional representations that preserve
important structure.

Generative models, including Variational Autoencoders (VAE) and Generative
Adversarial Networks (GAN), learn the underlying distribution of the training
data and can generate new, realistic samples.

## Reinforcement Learning

Reinforcement learning (RL) is inspired by behavioural psychology. An agent
interacts with an environment, receives rewards or penalties based on its actions,
and learns a policy that maximises cumulative reward. RL has achieved superhuman
performance in games like Chess, Go, and StarCraft II.

Key concepts in RL include the Markov Decision Process (MDP), Q-learning, and
policy gradient methods. Deep RL combines neural networks with RL to handle
high-dimensional state spaces, as demonstrated by AlphaGo and OpenAI Five.

## Deep Learning

Deep learning refers to neural networks with many layers (hence "deep"). These
networks automatically learn hierarchical feature representations from raw data.
Convolutional Neural Networks (CNNs) excel at image recognition. Recurrent Neural
Networks (RNNs) and their variants — LSTMs and GRUs — are suited for sequential
data like text and time-series.

Transformers, introduced in 2017, use self-attention mechanisms and have become
the dominant architecture for natural language processing tasks. Models like
BERT, GPT, and T5 have set state-of-the-art benchmarks across a wide range of
NLP benchmarks.

## Applications

Machine learning is applied across virtually every industry. In healthcare, ML
models detect cancer from medical images with accuracy rivalling expert clinicians.
In finance, algorithmic trading systems and fraud detection rely on ML. Autonomous
vehicles use deep learning for perception and planning. Recommendation systems
at Netflix, Spotify, and Amazon are powered by collaborative filtering and deep
learning models.

Natural language processing applications include machine translation, sentiment
analysis, question answering, and text summarisation. Computer vision applications
range from facial recognition to satellite image analysis.

## Challenges and Ethics

Despite its successes, machine learning faces significant challenges. Data quality
and quantity remain bottlenecks — models are only as good as the data they learn
from. Bias in training data leads to biased model predictions, raising serious
ethical concerns particularly in criminal justice and hiring.

Interpretability is another challenge: deep learning models are often described
as "black boxes" whose decisions are difficult to explain. Techniques like SHAP,
LIME, and attention visualisation attempt to address this. Robustness and security
are also concerns — adversarial examples can fool even highly accurate models.

The environmental cost of training large models is substantial; a single training
run of a large language model can emit as much CO₂ as five cars over their
lifetimes. Efficient architectures and green computing are active areas of research.

## Future Directions

The future of machine learning includes self-supervised learning, where models
learn from vast amounts of unlabelled data; few-shot and zero-shot learning that
generalise from minimal examples; and neuromorphic computing that mimics biological
neural processing. Foundation models trained on internet-scale data are becoming
general-purpose learners adaptable to many downstream tasks with minimal fine-tuning.
"""

DEMO_QUERY = "What are the main challenges and ethical concerns in machine learning?"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart Chunking & Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--doc",    type=str, help="Path to document (.txt/.json/.pdf) or raw text string")
    p.add_argument("--query",  type=str, help="Retrieval query string")
    p.add_argument("--demo",   action="store_true", help="Run with the built-in demo document and query")
    p.add_argument("--top-k",  type=int, default=5,   help="Number of top chunks to retrieve (default: 5)")
    p.add_argument("--semantic-threshold", type=float, default=0.15,
                   help="Cosine similarity threshold for semantic chunking (default: 0.15)")
    p.add_argument("--overlap-window", type=int, default=6,
                   help="Sentence window size for overlap chunking (default: 6)")
    p.add_argument("--overlap-size",   type=int, default=2,
                   help="Number of overlapping sentences (default: 2)")
    p.add_argument("--json-output", action="store_true",
                   help="Also dump the full result dict as JSON to stdout")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.demo:
        document = DEMO_DOCUMENT
        query    = DEMO_QUERY
        print("\n[Demo mode] Using built-in ML document.")
        print(f"[Demo mode] Query: {query!r}\n")
    elif args.doc and args.query:
        document = args.doc
        query    = args.query
    else:
        parser.print_help()
        print("\n⚠  Provide --doc and --query, or use --demo.\n")
        sys.exit(1)

    result = analyse(
        document=document,
        query=query,
        top_k=args.top_k,
        semantic_threshold=args.semantic_threshold,
        overlap_window=args.overlap_window,
        overlap_size=args.overlap_size,
        verbose=True,
    )

    if args.json_output:
        print("\n── Full JSON Output ──────────────────────────────────────────────────────")
        print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    main()
