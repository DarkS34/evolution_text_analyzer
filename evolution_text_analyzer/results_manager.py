"""
Enhanced results management module for medical diagnostic analysis.
Provides improved result handling for multi-model evaluations.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt

from .data_models import EvaluationResult


class ResultsManager:
    """Manages the storage, organization and visualization of model evaluation results."""

    def __init__(self, base_results_dir: Path, single_model_mode: bool):
        """
        Initialize the results manager.

        Args:
            base_results_dir: Base directory for storing results
        """
        self.base_dir = base_results_dir
        self.single_model_mode = single_model_mode
        self.results_dir = self._create_timestamped_results_dir()
        self.summary_data = []
        self.detailed_results = []

    def _create_timestamped_results_dir(self) -> Path:
        """Create a timestamped directory for the current evaluation run."""

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = self.base_dir / \
            f"{'individual_' if self.single_model_mode else ''}evaluation_run_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def add_result(self, result: EvaluationResult) -> None:
        """
        Add a model evaluation result and update the summary.

        Args:
            result: The evaluation result to add
        """
        # Add to detailed results
        self.detailed_results.append(result)

        # Extract summary info
        model_name = result.model_info.get("model_name", "Unknown")
        accuracy = result.performance.accuracy
        incorrect = result.performance.incorrect_outputs
        errors = result.performance.errors
        total_texts = result.performance.total_texts
        duration = result.performance.duration
        normalized = result.performance.normalized
        expanded = result.performance.expanded

        # Create summary entry
        summary_entry = {
            "model_name": model_name,
            "accuracy": accuracy,
            "incorrect_outputs": incorrect,
            "errors": errors,
            "total_texts": total_texts,
            "duration": duration,
            "normalized": normalized,
            "expanded": expanded,
        }

        self.summary_data.append(summary_entry)

        # Write individual result file
        self._write_individual_result(result)

        # Update summary file
        self._write_summary()

    def _write_individual_result(self, result: EvaluationResult) -> None:
        """
        Write an individual model result to its own file.

        Args:
            result: The evaluation result to write
        """

        model_name = result.model_info.get(
            "model_name", "unknown").replace(r"[:_]", "")
        normalized_tag = "_N" if result.performance.normalized else ""
        expanded_tag = "_E" if result.performance.expanded else ""

        # Change saving strategy if execute just for one model
        if self.single_model_mode:
            filename = f"detailed_results_{model_name}{normalized_tag}{expanded_tag}.json"
            file_path = self.results_dir / filename
        else:
            filename = f"{model_name}{normalized_tag}{expanded_tag}.json"
            file_path = self.results_dir / "individual_results" / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(exclude_none=True),
                      f, indent=2, ensure_ascii=False)

    def _write_summary(self) -> None:
        """Write the summary data to a JSON file and generate visualizations."""
        # Sort by accuracy (descending)
        sorted_summary = sorted(
            self.summary_data, key=lambda x: x["accuracy"], reverse=True)

        # Write JSON summary
        summary_path = self.results_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(sorted_summary, f, indent=2, ensure_ascii=False)

        # Create CSV for easier analysis
        df = pd.DataFrame(sorted_summary)
        csv_path = self.results_dir / "summary.csv"
        df.to_csv(csv_path, index=False)

        # Generate visualizations
        self._generate_visualizations(df)

    def _generate_visualizations(self, df: pd.DataFrame) -> None:
        """
        Generate visualization charts for model comparison.

        Args:
            df: DataFrame containing the summary data
        """
        if self.single_model_mode:
            viz_dir = self.results_dir
        else:
            viz_dir = self.results_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

        # Performance comparison chart
        plt.figure(figsize=(12, 8))
        df_sorted = df.sort_values("accuracy", ascending=False)
        if len(df_sorted) > 10:
            df_sorted = df_sorted.head(10)

        # Prepare data
        models = df_sorted["model_name"]
        accuracy = df_sorted["accuracy"]
        incorrect = df_sorted["incorrect_outputs"]
        errors = df_sorted["errors"]

        x = range(len(models))
        width = 0.25

        plt.bar([i - width for i in x], accuracy,
                width=width, label="Accuracy", color="green")
        plt.bar(x, incorrect, width=width, label="Incorrect", color="red")
        plt.bar([i + width for i in x], errors,
                width=width, label="Errors", color="orange")

        plt.xlabel("Models")
        plt.ylabel("Percentage (%)")
        plt.title("Model Performance Comparison")
        plt.xticks(x, models, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_comparison.png", dpi=300)
        plt.close()

        if not self.single_model_mode:
            plt.figure(figsize=(12, 6))

            df_time_sorted = df.sort_values("duration")

            if len(df_time_sorted) > 10:
                df_time_sorted = df_time_sorted.head(10)

            plt.barh(df_time_sorted["model_name"], df_time_sorted["duration"])
            plt.xlabel("Execution Time (seconds)")
            plt.title("Model Execution Time Comparison")
            plt.tight_layout()
            plt.savefig(viz_dir / "execution_time_comparison.png", dpi=300)
            plt.close()

    def get_best_model(self) -> Optional[Dict]:
        """
        Get the best performing model based on accuracy.

        Returns:
            Dictionary with the best model information or None if no models
        """
        if not self.summary_data:
            return None

        # Sort by accuracy (descending) and return the first one
        sorted_models = sorted(
            self.summary_data, key=lambda x: x["accuracy"], reverse=True)
        return sorted_models[0]

    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive text report of the evaluation results.

        Returns:
            Path to the generated report file
        """
        report_path = self.results_dir / "evaluation_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MEDICAL DIAGNOSTIC MODELS EVALUATION REPORT\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")

            if self.summary_data:
                best_model = self.get_best_model()

                f.write(f"Total models evaluated: {len(self.summary_data)}\n")
                f.write(
                    f"Best performing model: {best_model['model_name']} (Accuracy: {best_model['accuracy']}%)\n")
                f.write(
                    f"Average accuracy across all models: {sum(m['accuracy'] for m in self.summary_data) / len(self.summary_data):.2f}%\n\n")

                f.write("MODEL COMPARISON\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"{'Model Name':<30} {'Accuracy':<10} {'Incorrect':<10} {'Errors':<10} {'Duration':<10}\n")
                f.write("-" * 80 + "\n")

                sorted_models = sorted(
                    self.summary_data, key=lambda x: x["accuracy"], reverse=True)

                for model in sorted_models:
                    f.write(
                        f"{model['model_name']:<30} {model['accuracy']:<10.2f} {model['incorrect_outputs']:<10.2f} {model['errors']:<10.2f} {model['duration']:<10.2f}\n")
            else:
                f.write("No models evaluated.\n")

        return str(report_path)
