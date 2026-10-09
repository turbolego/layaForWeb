#!/usr/bin/env python3
"""Train a Laya model on WCAG 2.2 accessibility violation detection.

    # From Kaggle dataset\n",
    "python scripts/train_wcag.py --dataset wcag_train.json\n",
    "# Custom dataset\n",
    "python scripts/train_wcag.py --dataset /path/to/dataset.json --output build/wcag\n",
    "# With custom parameters\n",
    "python scripts/train_wcag.py --epochs 5 --batch_size 8 --lr 5e-5"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import laya
from laya import LayaDataset, LayaModel


def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    if not dataset:
        raise ValueError("Empty dataset")
    
    texts = [sample["text"] for sample in dataset if sample.get("text")]
    questions = [sample.get("questions", {}) for sample in dataset]
    labels = [sample.get("labels", {}) for sample in dataset]
    
    if len(texts) != len(questions) or len(texts) != len(labels):
        raise ValueError(f"Dataset size mismatch: {len(texts)} texts, {len(questions)} questions, {len(labels)} labels")
    
    return texts, questions, labels


def main():
    parser = argparse.ArgumentParser(description="Train Laya model for WCAG 2.2 violation detection")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", default="build/wcag", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--model", default="convaiinnovations/laya", help="Base Laya model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset}")
    texts, questions, labels = load_dataset(args.dataset)
    print(f"Loaded {len(texts)} samples")
    
    # Create Laya dataset
    dataset = LayaDataset(texts=texts, questions=questions, labels=labels)
    
    print(f"Loading model: {args.model}")
    agent = laya.load(args.model, device=args.device)
    print(f"Model loaded with {sum(p.numel() for p in agent.model.parameters()):,} parameters")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    
    agent.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True
    )
    
    # Evaluate
    loss, accuracy = agent.evaluate(dataset)
    print(f"\nEvaluation:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs(args.output, exist_ok=True)
    agent.save(args.output)
    print(f"\nModel saved to {args.output}")
    
    # List saved files
    print(f"\nFiles:")
    for root, dirs, files in os.walk(args.output):
        level = root.replace(args.output, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')


if __name__ == "__main__":
    main()
