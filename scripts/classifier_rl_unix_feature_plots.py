import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import numpy as np
import pandas as pd
import seaborn as sns


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
        return ''.join(random.choices(alphabet, k=n))
    
    start = random.randint(0, len(base)-n)
    return ''.join([c if random.random() > 0.4 else random.choice(alphabet) 
                    for c in base[start:start+n]])


def classify_sequence(sequence, detectors, n, r=0):
    """Computes anomaly score based on unmatched chunks."""
    if len(sequence) < n:
        return 1.0

    chunks = [sequence[i:i+n] for i in range(len(sequence) - n + 1)]
    unmatched_count = sum(1 for chunk in chunks if not any(hamming_distance(chunk, d) <= r for d in detectors))
    return unmatched_count / len(chunks) if chunks else 1.0


def manage_detector_set_size_batch_based(detector_batches, target_size, test_sequences, labels, n, r):
    """Manage detector set size by evaluating and removing least effective batches."""
    if not detector_batches:
        return set(), []
    
    total_detectors = sum(len(batch[1]) for batch in detector_batches)
    
    if total_detectors <= target_size:
        combined = set()
        for _, batch_detectors, _ in detector_batches:
            combined.update(batch_detectors)
        return combined, detector_batches
    
    # Evaluate each batch's contribution
    batch_contributions = []
    
    for i, (batch_id, batch_detectors, method_name) in enumerate(detector_batches):
        without_batch = set()
        for j, (_, other_batch, _) in enumerate(detector_batches):
            if i != j:
                without_batch.update(other_batch)
        
        with_batch = without_batch.union(batch_detectors)
        
        if without_batch:
            scores_without = [classify_sequence(seq, without_batch, n, r) for seq in test_sequences]
            auc_without = roc_auc_score(labels, scores_without)
        else:
            auc_without = 0.5
        
        scores_with = [classify_sequence(seq, with_batch, n, r) for seq in test_sequences]
        auc_with = roc_auc_score(labels, scores_with)
        
        contribution = auc_with - auc_without
        batch_contributions.append((i, contribution, len(batch_detectors), method_name))
    
    batch_contributions.sort(key=lambda x: x[1] / x[2] if x[2] > 0 else 0, reverse=True)
    
    selected_batches = []
    current_size = 0
    
    for batch_idx, contribution, batch_size, method_name in batch_contributions:
        if current_size + batch_size <= target_size:
            selected_batches.append(detector_batches[batch_idx])
            current_size += batch_size
        elif current_size < target_size:
            remaining_space = target_size - current_size
            batch_id, batch_detectors, method_name = detector_batches[batch_idx]
            partial_batch = set(random.sample(list(batch_detectors), remaining_space))
            selected_batches.append((batch_id, partial_batch, method_name))
            break
    
    combined = set()
    for _, batch_detectors, _ in selected_batches:
        combined.update(batch_detectors)
    
    return combined, selected_batches


def should_replace_worst_batch(current_batches, new_batch, test_sequences, labels, n, r):
    """Determine if new batch should replace the worst performing existing batch."""
    if not current_batches:
        return True, -1, float('inf')
    
    current_combined = set()
    for _, batch_detectors, _ in current_batches:
        current_combined.update(batch_detectors)
    
    current_scores = [classify_sequence(seq, current_combined, n, r) for seq in test_sequences]
    current_auc = roc_auc_score(labels, current_scores)
    
    worst_batch_idx = -1
    worst_contribution = float('inf')
    
    for i, (batch_id, batch_detectors, method_name) in enumerate(current_batches):
        without_batch = set()
        for j, (_, other_batch, _) in enumerate(current_batches):
            if i != j:
                without_batch.update(other_batch)
        
        if without_batch:
            scores_without = [classify_sequence(seq, without_batch, n, r) for seq in test_sequences]
            auc_without = roc_auc_score(labels, scores_without)
            contribution = current_auc - auc_without
        else:
            contribution = current_auc - 0.5
        
        if contribution < worst_contribution:
            worst_contribution = contribution
            worst_batch_idx = i
    
    test_combined = set()
    for j, (_, other_batch, _) in enumerate(current_batches):
        if j != worst_batch_idx:
            test_combined.update(other_batch)
    test_combined.update(new_batch)
    
    test_scores = [classify_sequence(seq, test_combined, n, r) for seq in test_sequences]
    test_auc = roc_auc_score(labels, test_scores)
    
    improvement = test_auc - current_auc
    should_replace = improvement > 0.001
    
    return should_replace, worst_batch_idx, improvement


def calculate_detailed_features(detector_batches, all_self, alphabet, n, r):
    """Calculate detailed feature metrics for tracking evolution."""
    if not detector_batches:
        return {
            'diversity': 0, 'coverage': 0, 'false_positive_rate': 0, 
            'uniformity': 0, 'total_detectors': 0, 'unique_detectors': 0,
            'batch_diversity': 0, 'method_balance': 0
        }
    
    # Combine all detectors from batches
    all_detectors = set()
    for _, batch_detectors, _ in detector_batches:
        all_detectors.update(batch_detectors)
    
    if not all_detectors:
        return {
            'diversity': 0, 'coverage': 0, 'false_positive_rate': 0,
            'uniformity': 0, 'total_detectors': 0, 'unique_detectors': 0,
            'batch_diversity': 0, 'method_balance': 0
        }
    
    detectors_list = list(all_detectors)
    total_detectors = sum(len(batch[1]) for batch in detector_batches)
    unique_detectors = len(all_detectors)
    
    # Diversity: average hamming distance between detectors
    if len(detectors_list) > 1:
        sample_size = min(100, len(detectors_list))
        detector_samples = random.sample(detectors_list, sample_size)
        avg_distance = sum(hamming_distance(d1, d2) 
                        for i, d1 in enumerate(detector_samples) 
                        for d2 in detector_samples[i+1:]) / (sample_size * (sample_size - 1) / 2)
        diversity = avg_distance / n
    else:
        diversity = 0
    
    # Coverage: ability to detect non-self strings
    sample_size = 100
    random_strings = [''.join(random.choices(alphabet, k=n)) for _ in range(sample_size)]
    non_self_strings = [s for s in random_strings if s not in all_self]
    if non_self_strings:
        coverage = sum(any(hamming_distance(s, d) <= r for d in all_detectors) 
                    for s in non_self_strings) / len(non_self_strings)
    else:
        coverage = 0
    
    # False positive rate: matching self strings
    sample_size = min(100, len(all_self))
    if sample_size > 0:
        self_samples = random.sample(list(all_self), sample_size)
        false_positive_rate = sum(any(hamming_distance(s, d) <= r for d in all_detectors)
                                for s in self_samples) / sample_size
    else:
        false_positive_rate = 0
    
    # Uniformity: how uniform the distance distribution is
    if len(detectors_list) > 1:
        distances = [hamming_distance(d1, d2) 
                    for i, d1 in enumerate(detector_samples) 
                    for d2 in detector_samples[i+1:]]
        uniformity = 1.0 - (np.std(distances) / n if distances else 0)
    else:
        uniformity = 0
    
    # Batch-level diversity: how different batches are from each other
    if len(detector_batches) > 1:
        batch_overlaps = []
        for i in range(len(detector_batches)):
            for j in range(i+1, len(detector_batches)):
                batch1 = detector_batches[i][1]
                batch2 = detector_batches[j][1]
                overlap = len(batch1.intersection(batch2)) / min(len(batch1), len(batch2))
                batch_overlaps.append(1 - overlap)  # 1 - overlap = diversity
        batch_diversity = np.mean(batch_overlaps) if batch_overlaps else 0
    else:
        batch_diversity = 0
    
    # Method balance: how balanced the generation methods are
    methods = [method for _, _, method in detector_batches]
    method_counts = {method: methods.count(method) for method in set(methods)}
    if len(method_counts) > 1:
        # Calculate entropy-based balance (higher = more balanced)
        total = len(methods)
        entropy = -sum((count/total) * np.log2(count/total) for count in method_counts.values())
        max_entropy = np.log2(len(method_counts))
        method_balance = entropy / max_entropy if max_entropy > 0 else 0
    else:
        method_balance = 0
    
    return {
        'diversity': diversity,
        'coverage': coverage,
        'false_positive_rate': false_positive_rate,
        'uniformity': uniformity,
        'total_detectors': total_detectors,
        'unique_detectors': unique_detectors,
        'batch_diversity': batch_diversity,
        'method_balance': method_balance
    }


def plot_feature_evolution(feature_history, output_prefix="feature_evolution"):
    """Create comprehensive plots showing feature evolution over episodes."""
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(feature_history)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Feature Evolution Over Training Episodes', fontsize=16, fontweight='bold')
    
    # Plot each feature
    features_to_plot = [
        ('diversity', 'Detector Diversity', 'Avg Hamming Distance / n'),
        ('coverage', 'Coverage', 'Fraction of Non-Self Detected'),
        ('false_positive_rate', 'False Positive Rate', 'Fraction of Self Detected'),
        ('uniformity', 'Distance Uniformity', '1 - std(distances) / n'),
        ('batch_diversity', 'Batch Diversity', 'Avg Inter-Batch Diversity'),
        ('method_balance', 'Method Balance', 'Entropy-based Balance'),
        ('auc', 'Performance (AUC)', 'Area Under ROC Curve'),
        ('unique_detectors', 'Unique Detectors', 'Number of Unique Detectors'),
        ('total_detectors', 'Total Detectors', 'Total Detector Count')
    ]
    
    for idx, (feature, title, ylabel) in enumerate(features_to_plot):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        if feature in df.columns:
            ax.plot(df['episode'], df[feature], linewidth=2, alpha=0.8)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df['episode'], df[feature], 1)
            p = np.poly1d(z)
            ax.plot(df['episode'], p(df['episode']), "r--", alpha=0.8, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    feature_cols = ['diversity', 'coverage', 'false_positive_rate', 'uniformity', 
                   'batch_diversity', 'method_balance', 'auc', 'unique_detectors']
    correlation_matrix = df[feature_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create action choice evolution plot
    if 'action' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Calculate rolling average of action choices
        window_size = 10
        action_counts = []
        episodes = []
        
        for i in range(window_size, len(df)):
            window_data = df.iloc[i-window_size:i]
            action_dist = window_data['action'].value_counts(normalize=True).sort_index()
            action_counts.append([action_dist.get(j, 0) for j in range(3)])
            episodes.append(df.iloc[i]['episode'])
        
        action_counts = np.array(action_counts)
        action_names = ['Random', 'Mutation', 'Coverage']
        
        for i, name in enumerate(action_names):
            ax.plot(episodes, action_counts[:, i], label=name, linewidth=2)
        
        ax.set_title('Action Selection Evolution (Rolling Average)', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Action Selection Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_actions.png', dpi=300, bbox_inches='tight')
        plt.close()


def train_rl_detector_system(train_sequences, test_sequences, labels, n=4, r=1, episodes=200, 
                           target_detectors=2000, batch_size=100, max_batches=21):
    # Extract alphabet and self strings
    alphabet = list({c for seq in train_sequences for c in seq})
    all_self = {seq[i:i+n] for seq in train_sequences for i in range(len(seq)-n+1)}
    
    # RL setup
    state_size = 4
    action_size = 3
    Q = np.zeros((state_size, action_size))
    
    # Learning parameters
    alpha = 0.05
    gamma = 0.9
    epsilon = 0.1
    
    # Tracking variables
    detector_batches = []
    batch_counter = 0
    best_auc = 0.0
    best_detector_batches = []
    
    # Feature evolution tracking
    feature_history = []
    
    def get_detector_state_features(detector_batches, all_self):
        """Extract state features for RL (simplified version for state space)."""
        features = calculate_detailed_features(detector_batches, all_self, alphabet, n, r)
        
        # Discretize features for state space
        diversity = min(int(features['diversity'] * 10), 9)
        coverage = min(int(features['coverage'] * 10), 9) 
        false_positive_rate = min(int(features['false_positive_rate'] * 10), 9)
        uniformity = min(int(features['uniformity'] * 10), 9)
        
        state_index = (diversity + coverage * 10 + false_positive_rate * 100 + uniformity * 1000) % state_size
        return state_index
    
    # Detector generation functions
    def generate_random_detectors_batch(num_detectors=batch_size):
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
                all_existing = existing_detectors.union(detectors)
                if not all_existing or sum(1 for d in all_existing if hamming_distance(candidate, d) <= r) / len(all_existing) < 0.15:
                    detectors.add(candidate)
        return detectors
    
    # Training loop with feature tracking
    for episode in range(episodes):
        state = get_detector_state_features(detector_batches, all_self)
        
        # Choose action (epsilon-greedy)
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            action = np.argmax(Q[state])
        
        # Generate new batch
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
        
        # Add or replace batch
        if len(detector_batches) < max_batches:
            detector_batches.append((batch_counter, new_batch, method_name))
            batch_counter += 1
        else:
            should_replace, worst_idx, improvement = should_replace_worst_batch(
                detector_batches, new_batch, test_sequences, labels, n, r
            )
            if should_replace:
                detector_batches[worst_idx] = (batch_counter, new_batch, method_name)
                batch_counter += 1
        
        # Manage detector set size
        combined_detectors, detector_batches = manage_detector_set_size_batch_based(
            detector_batches, target_detectors, test_sequences, labels, n, r
        )
        
        # Evaluate performance
        scores = [classify_sequence(seq, combined_detectors, n, r) for seq in test_sequences]
        auc = roc_auc_score(labels, scores)
        
        # Calculate reward and update Q-table
        reward = auc - best_auc if auc > best_auc else (auc - best_auc) * 0.1
        if auc > best_auc:
            best_auc = auc
            best_detector_batches = detector_batches.copy()
        
        next_state = get_detector_state_features(detector_batches, all_self)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        # Track feature evolution
        features = calculate_detailed_features(detector_batches, all_self, alphabet, n, r)
        features.update({
            'episode': episode,
            'auc': auc,
            'action': action,
            'method': method_name,
            'reward': reward,
            'epsilon': epsilon,
            'best_auc': best_auc
        })
        feature_history.append(features)
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)
        
        # Progress reporting
        if episode % 10 == 0:
            print(f"Episode {episode}: AUC = {auc:.4f}, Best = {best_auc:.4f}, "
                  f"Diversity = {features['diversity']:.3f}, Coverage = {features['coverage']:.3f}")
    
    # Generate feature evolution plots
    plot_feature_evolution(feature_history, f"feature_evolution_{episodes}episodes")
    
    # Save feature history for further analysis
    pd.DataFrame(feature_history).to_csv(f'feature_history_{episodes}episodes.csv', index=False)
    
    # Convert best batches to final detector set
    best_combined_detectors = set()
    for _, batch_detectors, _ in best_detector_batches:
        best_combined_detectors.update(batch_detectors)
    
    print(f"\nTraining completed. Best AUC: {best_auc:.4f}")
    print(f"Final detector set size: {len(best_combined_detectors)}")
    print(f"Feature evolution plots saved as 'feature_evolution_{episodes}episodes_*.png'")
    print(f"Feature history saved as 'feature_history_{episodes}episodes.csv'")
    
    return Q, best_combined_detectors, feature_history


if __name__ == "__main__":
    dataset_folder = "syscalls/snd-cert"
    
    # Load data
    train_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.train")
    train_sequences = preprocess_file(train_file)
    
    test_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.1.test")
    labels_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.1.labels")
    
    test_sequences = preprocess_file(test_file)
    labels = [int(label) for label in preprocess_file(labels_file)]
    
    # Train RL system with feature tracking
    Q, best_detectors, feature_history = train_rl_detector_system(
        train_sequences, test_sequences, labels, 
        n=4, r=1, episodes=50, target_detectors=2000, 
        batch_size=100, max_batches=21
    )
    
    # Evaluate on other test sets
    for i in range(2, 4):
        test_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.{i}.test")
        labels_file = os.path.join(dataset_folder, f"{os.path.basename(dataset_folder)}.{i}.labels")
        
        test_sequences = preprocess_file(test_file)
        labels = [int(label) for label in preprocess_file(labels_file)]
        
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
