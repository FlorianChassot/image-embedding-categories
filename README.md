# image-embedding-categories

This project explores unsupervised image clustering using semantic embeddings — with the goal of discovering meaningful visual categories without predefined labels.

## What It Does

- Extracts embeddings from images using ViT / CLIP
- Clusters images (K-Means) based on visual similarity

- Tests alignment with existing categories (e.g. Reddit subreddits)

## Current Focus

- Fetching images from 8 Reddit subreddits using arctic_shift

- Embedding + clustering shows strong grouping (up to 99% match)

- CLIP-based naming improves cluster clarity

- Exploring a pipeline to auto-label clusters via LLMs

## Stack

- Python

- HuggingFace Transformers (ViT / CLIP)

- scikit-learn (KMeans)

- Reddit API via arctic_shift

## Status

Personal learning project – work in progress!