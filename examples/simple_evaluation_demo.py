#!/usr/bin/env python3
"""
Simple demo showing the easiest ways to use the SAR dataset evaluators.

This script demonstrates three levels of complexity:
1. BASIC: Use all defaults - just specify datasets
2. INTERMEDIATE: Use defaults with custom parameters
3. ADVANCED: Add custom path generators alongside defaults
"""

import os
import sarenv
from sarenv.analytics.evaluator import ComparativeEvaluator, ComparativeDatasetEvaluator

log = sarenv.get_logger()


def demo_basic_usage():
    """Simplest possible usage - just specify the dataset directories."""
    
    print("\n" + "="*60)
    print("DEMO 1: BASIC Usage (All Defaults)")
    print("="*60)
    
    # All you need - the evaluator will use sensible defaults
    evaluator = ComparativeDatasetEvaluator(
        dataset_dirs=["sarenv_dataset/1", "sarenv_dataset/2"],
        num_victims=50,  # Small number for quick demo
        evaluation_size="small"
    )
    
    print(f"Default path generators: {list(evaluator.path_generators.keys())}")
    print("Running evaluation with all defaults...")
    
    # Run the evaluation
    results = evaluator.evaluate()
    summary = evaluator.summarize_results()
    print("\nResults summary:")
    print(summary.to_string(index=False))
    
    return evaluator


def demo_intermediate_usage():
    """Customize drone parameters but keep default path generators."""
    
    print("\n" + "="*60)
    print("DEMO 2: INTERMEDIATE Usage (Custom Parameters)")
    print("="*60)
    
    # Customize drone/path parameters while keeping default generators
    evaluator = ComparativeDatasetEvaluator(
        dataset_dirs=["sarenv_dataset/1"],
        num_victims=30,
        evaluation_size="small",
        # Custom drone parameters
        num_drones=5,           # More drones
        fov_degrees=60.0,       # Wider field of view
        altitude_meters=100.0,  # Higher altitude
        overlap_ratio=0.3       # More overlap
    )
    
    print(f"Using {evaluator.num_drones} drones with {evaluator.fov_degrees}° FOV")
    print(f"Path generators: {list(evaluator.path_generators.keys())}")
    
    return evaluator


def demo_advanced_usage():
    """Add custom path generators alongside the defaults."""
    
    print("\n" + "="*60)
    print("DEMO 3: ADVANCED Usage (Custom + Default Generators)")
    print("="*60)
    
    # Get the default generators and add custom ones
    default_generators = ComparativeDatasetEvaluator.get_default_path_generators(
        num_drones=3,
        fov_degrees=45.0
    )
    
    # Add a custom generator
    from sarenv.analytics import paths
    
    custom_generators = default_generators.copy()
    custom_generators["CustomRandom"] = lambda cx, cy, r, item, max_length: paths.generate_random_walk_path(
        cx, cy, 3, item.heatmap, item.bounds, num_jumps=30  # More random jumps
    )
    
    evaluator = ComparativeDatasetEvaluator(
        dataset_dirs=["sarenv_dataset/1"],
        path_generators=custom_generators,  # Mix of default + custom
        num_victims=30,
        evaluation_size="small"
    )
    
    print(f"Available generators: {list(evaluator.path_generators.keys())}")
    print("Notice 'CustomRandom' is added to the defaults!")
    
    return evaluator


def demo_single_dataset_evaluator():
    """Show how to use the single-dataset ComparativeEvaluator."""
    
    print("\n" + "="*60)
    print("DEMO 4: Single Dataset Evaluator")
    print("="*60)
    
    # Single dataset evaluation with default generators
    single_evaluator = ComparativeEvaluator(
        dataset_directory="sarenv_dataset/1",
        # No need to specify baseline_generators - defaults will be used!
        num_lost_persons=20,
        evaluation_sizes=["small"]
    )
    
    print(f"Baseline generators: {list(single_evaluator.baseline_generators.keys())}")
    
    # Quick evaluation
    results = single_evaluator.run_baseline_evaluations()
    print(f"\nEvaluated {len(results)} scenarios")
    
    return single_evaluator


if __name__ == "__main__":
    print("SAR Dataset Evaluator - Simple Demo")
    print("This demo shows progressively more advanced usage patterns.")
    
    try:
        # Run demos
        basic_eval = demo_basic_usage()
        intermediate_eval = demo_intermediate_usage()
        advanced_eval = demo_advanced_usage()
        single_eval = demo_single_dataset_evaluator()
        
        print("\n" + "="*60)
        print("SUCCESS: All demos completed!")
        print("="*60)
        print("\nKey takeaways:")
        print("1. For basic use: Just specify dataset_dirs - everything else has defaults")
        print("2. For custom parameters: Specify drone/path parameters as needed")
        print("3. For custom algorithms: Get defaults and add your own generators")
        print("4. Single dataset evaluation: Use ComparativeEvaluator")
        print("\nAll evaluators come with these default path generators:")
        print("- RandomWalk/Random: Random walk search pattern")
        print("- Greedy: Greedy search focusing on high-probability areas")
        print("- Spiral: Systematic spiral coverage pattern")
        print("- Concentric: Concentric circles pattern")
        print("- Pizza: Pizza-slice zigzag pattern")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        log.error(f"Demo error: {e}")
        raise
