import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import numpy as np


def preprocess_file(file_path):
    """Reads a file and returns all sequences as a list of strings."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def hamming_distance(s1, s2):
    """Computes the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def mutate_substring(base, n, alphabet):
    """Creates candidates through controlled mutation."""
    if len(base) < n:
        # If base is too short for mutation with chunk size n, return a random string of length n
        return ''.join(random.choices(alphabet, k=n))
    
    # Otherwise, proceed with controlled mutation
    start = random.randint(0, len(base)-n)
    return ''.join([c if random.random() > 0.4 else random.choice(alphabet) 
                    for c in base[start:start+n]])


def classify_sequence(sequence, detectors, n, r=0):
    """Computes anomaly score based on unmatched chunks."""
    if len(sequence) < n:
        return 1.0  # Treat as anomalous if too short

    chunks = [sequence[i:i+n] for i in range(len(sequence) - n + 1)]
    unmatched_count = sum(1 for chunk in chunks if not any(hamming_distance(chunk, d) <= r for d in detectors))
    return unmatched_count / len(chunks) if chunks else 1.0


def manage_detector_set_size_batch_based(detector_batches, target_size, test_sequences, labels, n, r):
    """
    Manage detector set size by evaluating and removing least effective batches.
    
    Args:
        detector_batches: List of tuples (batch_id, detector_set, method_name)
        target_size: Target total number of detectors
        test_sequences: Test sequences for evaluation
        labels: True labels for evaluation
        n: Detector length
        r: Matching threshold
    
    Returns:
        Combined detector set and updated batch list
    """
    if not detector_batches:
        return set(), []
    
    # Calculate total current size
    total_detectors = sum(len(batch[1]) for batch in detector_batches)
    
    if total_detectors <= target_size:
        # Combine all batches if under target
        combined = set()
        for _, batch_detectors, _ in detector_batches:
            combined.update(batch_detectors)
        return combined, detector_batches
    
    # Evaluate each batch's contribution to overall performance
    batch_contributions = []
    
    for i, (batch_id, batch_detectors, method_name) in enumerate(detector_batches):
        # Create detector set without this batch
        without_batch = set()
        for j, (_, other_batch, _) in enumerate(detector_batches):
            if i != j:
                without_batch.update(other_batch)
        
        # Create detector set with this batch
        with_batch = without_batch.union(batch_detectors)
        
        # Evaluate performance difference
        if without_batch:
            scores_without = [classify_sequence(seq, without_batch, n, r) for seq in test_sequences]
            auc_without = roc_auc_score(labels, scores_without)
        else:
            auc_without = 0.5  # Random performance baseline
        
        scores_with = [classify_sequence(seq, with_batch, n, r) for seq in test_sequences]
        auc_with = roc_auc_score(labels, scores_with)
        
        # Calculate contribution (improvement in AUC)
        contribution = auc_with - auc_without
        batch_contributions.append((i, contribution, len(batch_detectors), method_name))
    
    # Sort batches by contribution per detector (efficiency metric)
    batch_contributions.sort(key=lambda x: x[1] / x[2] if x[2] > 0 else 0, reverse=True)
    
    # Select batches to keep based on target size
    selected_batches = []
    current_size = 0
    
    for batch_idx, contribution, batch_size, method_name in batch_contributions:
        if current_size + batch_size <= target_size:
            selected_batches.append(detector_batches[batch_idx])
            current_size += batch_size
        elif current_size < target_size:
            # Partially include the batch if there's remaining space
            remaining_space = target_size - current_size
            batch_id, batch_detectors, method_name = detector_batches[batch_idx]
            partial_batch = set(random.sample(list(batch_detectors), remaining_space))
            selected_batches.append((batch_id, partial_batch, method_name))
            break
    
    # Combine selected batches
    combined = set()
    for _, batch_detectors, _ in selected_batches:
        combined.update(batch_detectors)
    
    return combined, selected_batches


def should_replace_worst_batch(current_batches, new_batch, test_sequences, labels, n, r):
    """
    Determine if new batch should replace the worst performing existing batch.
    
    Returns:
        (should_replace, batch_to_replace_idx, performance_improvement)
    """
    if not current_batches:
        return True, -1, float('inf')
    
    # Evaluate current performance
    current_combined = set()
    for _, batch_detectors, _ in current_batches:
        current_combined.update(batch_detectors)
    
    current_scores = [classify_sequence(seq, current_combined, n, r) for seq in test_sequences]
    current_auc = roc_auc_score(labels, current_scores)
    
    # Find worst performing batch
    worst_batch_idx = -1
    worst_contribution = float('inf')
    
    for i, (batch_id, batch_detectors, method_name) in enumerate(current_batches):
        # Performance without this batch
        without_batch = set()
        for j, (_, other_batch, _) in enumerate(current_batches):
            if i != j:
                without_batch.update(other_batch)
        
        if without_batch:
            scores_without = [classify_sequence(seq, without_batch, n, r) for seq in test_sequences]
            auc_without = roc_auc_score(labels, scores_without)
            contribution = current_auc - auc_without
        else:
            contribution = current_auc - 0.5  # Baseline performance
        
        if contribution < worst_contribution:
            worst_contribution = contribution
            worst_batch_idx = i
    
    # Test performance with new batch replacing worst batch
    test_combined = set()
    for j, (_, other_batch, _) in enumerate(current_batches):
        if j != worst_batch_idx:
            test_combined.update(other_batch)
    test_combined.update(new_batch)
    
    test_scores = [classify_sequence(seq, test_combined, n, r) for seq in test_sequences]
    test_auc = roc_auc_score(labels, test_scores)
    
    improvement = test_auc - current_auc
    should_replace = improvement > 0.001  # Small threshold to avoid noise
    
    return should_replace, worst_batch_idx, improvement


def train_rl_detector_system(train_sequences, test_sequences, labels, n=4, r=1, episodes=200, 
                           target_detectors=2000, batch_size=400, max_batches=6):
    # Extract alphabet from training sequences
    alphabet = list({c for seq in train_sequences for c in seq})
    all_self = {seq[i:i+n] for seq in train_sequences for i in range(len(seq)-n+1)}
    
    # Define state features and actions
    state_size = 4
    action_size = 3
    
    # Initialize Q-table
    Q = np.zeros((state_size, action_size))
    
    # Learning parameters
    alpha = 0.05
    gamma = 0.9
    epsilon = 0.1
    
    # Track detector batches and performance
    detector_batches = []  # List of (batch_id, detector_set, method_name)
    batch_counter = 0
    best_auc = 0.0
    best_detector_batches = []
    
    def get_detector_state_features(detector_batches, all_self):
        """Extract meaningful state features from detector batches"""
        if not detector_batches:
            return 0
        
        # Combine all detectors from batches
        all_detectors = set()
        for _, batch_detectors, _ in detector_batches:
            all_detectors.update(batch_detectors)
        
        if not all_detectors:
            return 0
        
        detectors_list = list(all_detectors)
        
        # Calculate features
        if len(detectors_list) > 1:
            sample_size = min(100, len(detectors_list))
            detector_samples = random.sample(detectors_list, sample_size)
            avg_distance = sum(hamming_distance(d1, d2) 
                            for i, d1 in enumerate(detector_samples) 
                            for d2 in detector_samples[i+1:]) / (sample_size * (sample_size - 1) / 2)
            diversity = avg_distance / n
        else:
            diversity = 0
            
        sample_size = 100
        random_strings = [''.join(random.choices(alphabet, k=n)) for _ in range(sample_size)]
        non_self_strings = [s for s in random_strings if s not in all_self]
        if non_self_strings:
            coverage = sum(any(hamming_distance(s, d) <= r for d in all_detectors) 
                        for s in non_self_strings) / len(non_self_strings)
        else:
            coverage = 0
        
        sample_size = min(100, len(all_self))
        if sample_size > 0:
            self_samples = random.sample(list(all_self), sample_size)
            false_positive_rate = sum(any(hamming_distance(s, d) <= r for d in all_detectors)
                                    for s in self_samples) / sample_size
        else:
            false_positive_rate = 0
        
        if len(detectors_list) > 1:
            distances = [hamming_distance(d1, d2) 
                        for i, d1 in enumerate(detector_samples) 
                        for d2 in detector_samples[i+1:]]
            uniformity = 1.0 - (np.std(distances) / n if distances else 0)
        else:
            uniformity = 0
        
        state_vector = [diversity, coverage, false_positive_rate, uniformity]
        discretized = [min(int(f * 10), 9) for f in state_vector]
        state_index = sum(discretized[i] * (10 ** i) for i in range(len(discretized))) % state_size
        
        return state_index
    
    # Modified detector generation strategies
    def generate_random_detectors_batch(num_detectors=batch_size):
        """Generate a batch of random detectors"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            candidate = ''.join(random.choices(alphabet, k=n))
            if candidate not in all_self:
                detectors.add(candidate)
                
        return detectors
    
    def generate_mutation_based_detectors_batch(num_detectors=batch_size):
        """Generate a batch of mutation-based detectors"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            
            base = random.choice(train_sequences)
            if len(base) < n:
                continue
                
            start = random.randint(0, len(base)-n)
            candidate = ''.join([c if random.random() > 0.4 else random.choice(alphabet) 
                              for c in base[start:start+n]])
            
            if candidate not in all_self:
                detectors.add(candidate)
                
        return detectors
    
    def generate_coverage_optimized_detectors_batch(existing_detectors, num_detectors=batch_size):
        """Generate a batch of coverage-optimized detectors"""
        detectors = set()
        attempts = 0
        max_attempts = num_detectors * 10
        
        while len(detectors) < num_detectors and attempts < max_attempts:
            attempts += 1
            
            if random.random() < 0.3 and train_sequences:
                base = random.choice(train_sequences)
                if len(base) >= n:
                    start = random.randint(0, len(base)-n)
                    candidate = ''.join([c if random.random() > 0.4 else random.choice(alphabet) 
                                      for c in base[start:start+n]])
                else:
                    candidate = ''.join(random.choices(alphabet, k=n))
            else:
                candidate = ''.join(random.choices(alphabet, k=n))
            
            if candidate not in all_self and candidate not in existing_detectors:
                # Anti-clustering with existing detectors
                all_existing = existing_detectors.union(detectors)
                if not all_existing or sum(1 for d in all_existing if hamming_distance(candidate, d) <= r) / len(all_existing) < 0.15:
                    detectors.add(candidate)
                
        return detectors
    
    # Training loop
    for episode in range(episodes):
        # Current state: features of our detector batches
        state = get_detector_state_features(detector_batches, all_self)
        
        # Choose action (epsilon-greedy)
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            action = np.argmax(Q[state])
        
        # Generate detector batch based on action
        if action == 0:
            new_batch = generate_random_detectors_batch()
            method_name = "random"
        elif action == 1:
            new_batch = generate_mutation_based_detectors_batch()
            method_name = "mutation"
        else:
            current_combined = set()
            for _, batch_detectors, _ in detector_batches:
                current_combined.update(batch_detectors)
            new_batch = generate_coverage_optimized_detectors_batch(current_combined)
            method_name = "coverage"
        
        # Decide whether to add new batch
        if len(detector_batches) < max_batches:
            # Add new batch if we haven't reached maximum
            detector_batches.append((batch_counter, new_batch, method_name))
            batch_counter += 1
        else:
            # Check if new batch should replace worst existing batch
            should_replace, worst_idx, improvement = should_replace_worst_batch(
                detector_batches, new_batch, test_sequences, labels, n, r
            )
            
            if should_replace:
                print(f"Episode {episode}: Replacing batch {detector_batches[worst_idx][0]} "
                      f"({detector_batches[worst_idx][2]}) with new {method_name} batch "
                      f"(improvement: {improvement:.4f})")
                detector_batches[worst_idx] = (batch_counter, new_batch, method_name)
                batch_counter += 1
        
        # Manage total detector count across all batches
        combined_detectors, detector_batches = manage_detector_set_size_batch_based(
            detector_batches, target_detectors, test_sequences, labels, n, r
        )
        
        # Evaluate performance of combined detector set
        scores = [classify_sequence(seq, combined_detectors, n, r) for seq in test_sequences]
        auc = roc_auc_score(labels, scores)
        
        # Calculate reward based on AUC improvement
        reward = auc - best_auc if auc > best_auc else (auc - best_auc) * 0.1  # Penalty for worse performance
        
        # Update best detector set if improved
        if auc > best_auc:
            best_auc = auc
            best_detector_batches = detector_batches.copy()
            print(f"Episode {episode}: New best AUC = {best_auc:.4f}, Action = {action} ({method_name}), "
                  f"Batches = {len(detector_batches)}, Total detectors = {len(combined_detectors)}")
        
        # Update Q-table
        next_state = get_detector_state_features(detector_batches, all_self)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)
        
        # Print progress
        if episode % 50 == 0:
            batch_methods = [method for _, _, method in detector_batches]
            method_counts = {method: batch_methods.count(method) for method in set(batch_methods)}
            print(f"Episode {episode}: Current AUC = {auc:.4f}, Best AUC = {best_auc:.4f}")
            print(f"Active batches: {len(detector_batches)}, Total detectors: {len(combined_detectors)}")
            print(f"Batch composition: {method_counts}")
    
    # Convert best batches back to combined detector set for final return
    best_combined_detectors = set()
    for _, batch_detectors, _ in best_detector_batches:
        best_combined_detectors.update(batch_detectors)
    
    print(f"Training completed. Best AUC: {best_auc:.4f}")
    print(f"Final detector set size: {len(best_combined_detectors)}")
    print(f"Final batch composition: {[(batch_id, method, len(detectors)) for batch_id, detectors, method in best_detector_batches]}")
    
    return Q, best_combined_detectors


if __name__ == "__main__":
    dataset_folder = "syscalls/snd-cert"  # Change to desired dataset
    
    # Load data
    train_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.train")
    train_sequences = preprocess_file(train_file)
    
    # Use first test set for training the RL system
    test_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.1.test")
    labels_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.1.labels")
    
    test_sequences = preprocess_file(test_file)
    labels = [int(label) for label in preprocess_file(labels_file)]
    
    # Train RL system with batch-based management
    Q, best_detectors = train_rl_detector_system(
        train_sequences, test_sequences, labels, 
        n=4, r=1, episodes=50, target_detectors=2000, 
        batch_size=100, max_batches=21
    )
    
    # Evaluate on other test sets
    for i in range(2, 4):  # Test sets 2 and 3
        test_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.{i}.test")
        labels_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.{i}.labels")
        
        test_sequences = preprocess_file(test_file)
        labels = [int(label) for label in preprocess_file(labels_file)]
        
        # Evaluate using best detectors found by RL
        scores = [classify_sequence(seq, best_detectors, n=4, r=1) for seq in test_sequences]
        auc = roc_auc_score(labels, scores)
        print(f"Test set {i} AUC: {auc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (RL-optimized)")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("1-specificity (FPR)")
        plt.ylabel("Sensitivity (TPR)")
        plt.title(f"Receiver Operating Characteristic (AUC = {auc:.3f})")
        plt.legend()
        plt.savefig(f"{os.path.basename(dataset_folder)}_{i}_rl_optimized_roc_curve.png")
        plt.close()
