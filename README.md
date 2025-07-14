This project explores **unsupervised image clustering** using semantic embeddings — with the goal of discovering meaningful visual categories without predefined labels.

## ✨ New: First CLI Prototype Released

A first working version of the pipeline is now available and can be run via command line. 

### How to Use

⚠️ A Working OpenAI API key is required for the step 3 and 4 of the pipeline. 

1. **Prepare Input**  
   Place `.txt` files inside the input folder. Each file contains image URLs (one per line). 

2. **Run the Pipeline**

   ```bash
   python script.py
   ```

   This will:
   - Download all images
   - Extract embeddings (ViT)
   - Cluster them with size constraints
   - Sample and label entries using OpenAI
   - Auto-name clusters using GPT

3. **Optional CLI Flags**

   You can selectively run parts of the pipeline:

   ```bash
   python ImageCategorisation.py --embed --cluster --entry --label
   ```

   Or override parameters:

   ```bash
   python ImageCategorisation.py --clusters 10 --sample 30 --input "my_input" --output "results"
   ```

   Available options:
   - `--embed`: extract embeddings
   - `--cluster`: perform constrained K-Means
   - `--entry`: label individual images in each cluster
   - `--label`: assign a name to each cluster using LLM
   - `--input`, `--output`: set folders
   - `--clusters`, `--sample`, `--min-cluster-ratio`: adjust parameters

---

## What It Does

- Extracts embeddings from images using ViT or CLIP
- Clusters images using **constrained K-Means** (to balance cluster size)
- Auto-labels samples from each cluster using OpenAI
- Suggests human-readable cluster names via GPT

---

## Current Focus

- Determining the quality of the cluster naming.
- After acquiring the cluster names, possibly adding a multimodal model such as clip which will redo the categorisation using the obtained labels.
- Using the prototype to create categories from an unknown subset of pictures and getting insight from it.

---

## Stack

- Python
- HuggingFace Transformers (ViT / CLIP)
- `k-means-constrained` (for balanced clusters)
- OpenAI API (for GPT-based labeling)
- Arctic Shift (Reddit API wrapper)

---

## Status

> **First prototype is working!**  
This is a personal learning project and a work in progress.  
Pull requests, ideas, or feedback are welcome.
