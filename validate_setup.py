#!/usr/bin/env python3
"""
Setup Validation Script for Movie Recommendation System
=======================================================

This script validates that all components of the recommendation system
are properly installed and configured.

Author: Portfolio Project for Data Science Applications
"""

import sys
import os
import importlib
import subprocess
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_required_packages() -> List[Tuple[bool, str]]:
    """Check if all required packages are installed."""
    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'torch',
        'psutil'
    ]

    results = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            results.append((True, f"✓ {package}"))
        except ImportError:
            results.append((False, f"✗ {package} (not installed)"))

    return results

def check_project_structure() -> List[Tuple[bool, str]]:
    """Check if project structure is correct."""
    required_files = [
        'main.py',
        'run_experiment.py',
        'src/preprocess.py',
        'src/recommend.py',
        'src/collaborative_filtering.py',
        'src/evaluate.py',
        'src/utils.py',
        'tests/test_recommend.py',
        'requirements.txt',
        'config.json'
    ]

    results = []
    for file_path in required_files:
        if os.path.exists(file_path):
            results.append((True, f"✓ {file_path}"))
        else:
            results.append((False, f"✗ {file_path} (missing)"))

    return results

def check_data_files() -> List[Tuple[bool, str]]:
    """Check if data files are present."""
    data_files = [
        'data/processed_movies.csv.gz',
        'data/TMDB_movie_dataset_v11.csv'
    ]

    results = []
    for file_path in data_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            results.append((True, f"✓ {file_path} ({size_mb:.1f}MB)"))
        else:
            results.append((False, f"✗ {file_path} (missing)"))

    return results

def test_import_modules() -> List[Tuple[bool, str]]:
    """Test importing project modules."""
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

    modules = [
        ('src.preprocess', 'TMDBDataProcessor'),
        ('src.recommend', 'ContentBasedRecommender'),
        ('src.collaborative_filtering', 'CollaborativeFilteringRecommender'),
        ('src.evaluate', 'RecommendationEvaluator'),
        ('src.utils', 'ConfigManager')
    ]

    results = []
    for module_name, class_name in modules:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            results.append((True, f"✓ {module_name}.{class_name}"))
        except Exception as e:
            results.append((False, f"✗ {module_name}.{class_name} ({str(e)})"))

    return results

def run_basic_tests() -> Tuple[bool, str]:
    """Run basic unit tests."""
    try:
        # Change to project directory
        original_cwd = os.getcwd()
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)

        # Set PYTHONPATH and run tests
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'

        result = subprocess.run(
            [sys.executable, 'tests/test_recommend.py'],
            capture_output=True,
            text=True,
            env=env,
            timeout=60
        )

        os.chdir(original_cwd)

        if result.returncode == 0:
            # Count passed tests
            lines = result.stderr.split('\n')
            for line in lines:
                if 'OK' in line and 'Ran' in line:
                    return True, f"✓ Unit tests passed: {line.strip()}"
            return True, "✓ Unit tests passed"
        else:
            return False, f"✗ Unit tests failed: {result.stderr.split('FAILED')[0] if 'FAILED' in result.stderr else 'Unknown error'}"

    except Exception as e:
        return False, f"✗ Unit tests error: {str(e)}"

def main():
    """Main validation function."""
    print("🎬 Movie Recommendation System - Setup Validation")
    print("=" * 60)

    all_passed = True

    # Check Python version
    passed, message = check_python_version()
    print(f"\nPython Version: {message}")
    if not passed:
        all_passed = False

    # Check required packages
    print(f"\nRequired Packages:")
    package_results = check_required_packages()
    for passed, message in package_results:
        print(f"  {message}")
        if not passed:
            all_passed = False

    # Check project structure
    print(f"\nProject Structure:")
    structure_results = check_project_structure()
    for passed, message in structure_results:
        print(f"  {message}")
        if not passed:
            all_passed = False

    # Check data files
    print(f"\nData Files:")
    data_results = check_data_files()
    for passed, message in data_results:
        print(f"  {message}")
        if not passed:
            all_passed = False

    # Test module imports
    print(f"\nModule Imports:")
    import_results = test_import_modules()
    for passed, message in import_results:
        print(f"  {message}")
        if not passed:
            all_passed = False

    # Run basic tests
    print(f"\nUnit Tests:")
    test_passed, test_message = run_basic_tests()
    print(f"  {test_message}")
    if not test_passed:
        all_passed = False

    # Summary
    print(f"\n" + "=" * 60)
    if all_passed:
        print("🎉 SUCCESS: All validation checks passed!")
        print("🚀 Your Movie Recommendation System is ready to use!")
        print("\nQuick start commands:")
        print("  python main.py --quick-demo --movie 'The Dark Knight'")
        print("  python run_experiment.py --experiments content_based")
        print("  PYTHONPATH=. python tests/test_recommend.py")
    else:
        print("❌ FAILED: Some validation checks failed.")
        print("📋 Please fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("  pip install -r requirements.txt")
        print("  Make sure data files are in the correct location")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())