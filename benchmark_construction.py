"""
Hallucination Benchmark Dataset Construction and Analysis
Authors: Jasmaine Khale & Saahithi Mallarapu

Complete toolkit for creating, managing, annotating, and analyzing
a controlled hallucination benchmark dataset for LLM evaluation.

Usage:
    python hallucination_benchmark.py --create          # Create sample benchmark
    python hallucination_benchmark.py --annotate        # Start annotation session
    python hallucination_benchmark.py --analyze         # Analyze results
    python hallucination_benchmark.py --agreement       # Calculate inter-annotator agreement
"""

import json
import csv
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict


# ============================================================================
# DATA MODELS
# ============================================================================

class QuestionCategory(Enum):
    """Categories of questions in the benchmark"""
    FACTUAL = "factual"
    REASONING = "reasoning"
    MULTISTEP = "multistep"
    DOMAIN_SPECIFIC = "domain_specific"


class Domain(Enum):
    """Domain-specific categories"""
    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    HISTORICAL = "historical"


@dataclass
class Question:
    """Represents a single question in the benchmark"""
    id: str
    category: str
    domain: str
    question_text: str
    ground_truth: str
    difficulty: int  # 1-5 scale
    requires_calculation: bool
    requires_external_knowledge: bool
    notes: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class Annotation:
    """Represents an annotation for a model response"""
    response_id: str
    question_id: str
    strategy: str
    annotator_id: str
    has_hallucination: bool
    factual_correctness: int  # 0-5 scale
    completeness: int  # 0-5 scale
    hallucination_details: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ModelResponse:
    """Represents a model's response to a question"""
    response_id: str
    question_id: str
    strategy: str
    model_name: str
    response_text: str
    generation_time: float
    cost: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ============================================================================
# BENCHMARK DATASET MANAGER
# ============================================================================

class BenchmarkDataset:
    """Main class for managing the benchmark dataset"""
    
    def __init__(self, name: str = "hallucination_benchmark"):
        self.name = name
        self.questions: List[Question] = []
        self.responses: List[ModelResponse] = []
        self.annotations: List[Annotation] = []
    
    def add_question(self, question: Question) -> None:
        """Add a question to the benchmark"""
        self.questions.append(question)
    
    def add_response(self, response: ModelResponse) -> None:
        """Add a model response"""
        self.responses.append(response)
    
    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation"""
        self.annotations.append(annotation)
    
    def get_questions_by_category(self, category: QuestionCategory) -> List[Question]:
        """Retrieve questions by category"""
        return [q for q in self.questions if q.category == category.value]
    
    def get_questions_by_domain(self, domain: Domain) -> List[Question]:
        """Retrieve questions by domain"""
        return [q for q in self.questions if q.domain == domain.value]
    
    def get_responses_for_question(self, question_id: str) -> List[ModelResponse]:
        """Get all responses for a specific question"""
        return [r for r in self.responses if r.question_id == question_id]
    
    def get_annotations_for_response(self, response_id: str) -> List[Annotation]:
        """Get all annotations for a specific response"""
        return [a for a in self.annotations if a.response_id == response_id]
    
    def save_to_json(self, filepath: str) -> None:
        """Save the entire benchmark to JSON"""
        data = {
            "name": self.name,
            "created_at": datetime.now().isoformat(),
            "questions": [asdict(q) for q in self.questions],
            "responses": [asdict(r) for r in self.responses],
            "annotations": [asdict(a) for a in self.annotations]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Benchmark saved to {filepath}")
    
    def load_from_json(self, filepath: str) -> None:
        """Load benchmark from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.name = data.get("name", self.name)
        self.questions = [Question(**q) for q in data.get("questions", [])]
        self.responses = [ModelResponse(**r) for r in data.get("responses", [])]
        self.annotations = [Annotation(**a) for a in data.get("annotations", [])]
        
        print(f"✓ Loaded {len(self.questions)} questions, {len(self.responses)} responses, {len(self.annotations)} annotations")
    
    def export_questions_csv(self, filepath: str) -> None:
        """Export questions to CSV for easy viewing/editing"""
        if not self.questions:
            print("⚠ No questions to export")
            return
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.questions[0]).keys())
            writer.writeheader()
            for question in self.questions:
                writer.writerow(asdict(question))
        
        print(f"✓ Questions exported to {filepath}")
    
    def import_questions_csv(self, filepath: str) -> None:
        """Import questions from CSV"""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['requires_calculation'] = row['requires_calculation'].lower() == 'true'
                row['requires_external_knowledge'] = row['requires_external_knowledge'].lower() == 'true'
                row['difficulty'] = int(row['difficulty'])
                self.questions.append(Question(**row))
        
        print(f"✓ Imported {len(self.questions)} questions from {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get benchmark statistics"""
        stats = {
            "total_questions": len(self.questions),
            "total_responses": len(self.responses),
            "total_annotations": len(self.annotations),
            "questions_by_category": {},
            "questions_by_domain": {},
            "questions_by_difficulty": {}
        }
        
        for q in self.questions:
            stats["questions_by_category"][q.category] = \
                stats["questions_by_category"].get(q.category, 0) + 1
            stats["questions_by_domain"][q.domain] = \
                stats["questions_by_domain"].get(q.domain, 0) + 1
            stats["questions_by_difficulty"][q.difficulty] = \
                stats["questions_by_difficulty"].get(q.difficulty, 0) + 1
        
        return stats


# ============================================================================
# QUESTION GENERATOR
# ============================================================================

class QuestionGenerator:
    """Helper class for generating sample questions"""
    
    @staticmethod
    def generate_factual_questions(count: int = 25) -> List[Question]:
        """Generate sample factual questions"""
        questions = []
        
        templates = [
            {
                "text": "What year did the Berlin Wall fall?",
                "truth": "1989",
                "domain": Domain.HISTORICAL,
                "difficulty": 2
            },
            {
                "text": "What is the capital of Australia?",
                "truth": "Canberra",
                "domain": Domain.GENERAL,
                "difficulty": 2
            },
            {
                "text": "Who wrote 'To Kill a Mockingbird'?",
                "truth": "Harper Lee",
                "domain": Domain.GENERAL,
                "difficulty": 1
            },
            {
                "text": "What is the chemical formula for water?",
                "truth": "H2O",
                "domain": Domain.SCIENTIFIC,
                "difficulty": 1
            },
            {
                "text": "When was the first iPhone released?",
                "truth": "June 29, 2007",
                "domain": Domain.TECHNICAL,
                "difficulty": 2
            },
            {
                "text": "What is the speed of light in vacuum?",
                "truth": "299,792,458 meters per second (approximately 300,000 km/s)",
                "domain": Domain.SCIENTIFIC,
                "difficulty": 2
            },
            {
                "text": "Who painted the Mona Lisa?",
                "truth": "Leonardo da Vinci",
                "domain": Domain.GENERAL,
                "difficulty": 1
            },
            {
                "text": "What year did World War II end?",
                "truth": "1945",
                "domain": Domain.HISTORICAL,
                "difficulty": 1
            },
            {
                "text": "What is the largest planet in our solar system?",
                "truth": "Jupiter",
                "domain": Domain.SCIENTIFIC,
                "difficulty": 1
            },
            {
                "text": "Who was the first person to walk on the moon?",
                "truth": "Neil Armstrong",
                "domain": Domain.HISTORICAL,
                "difficulty": 1
            }
        ]
        
        for i, template in enumerate(templates[:count]):
            questions.append(Question(
                id=f"factual_{i+1:03d}",
                category=QuestionCategory.FACTUAL.value,
                domain=template["domain"].value,
                question_text=template["text"],
                ground_truth=template["truth"],
                difficulty=template["difficulty"],
                requires_calculation=False,
                requires_external_knowledge=True,
                notes="Sample factual question"
            ))
        
        return questions
    
    @staticmethod
    def generate_reasoning_questions(count: int = 25) -> List[Question]:
        """Generate sample reasoning questions"""
        questions = []
        
        templates = [
            {
                "text": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                "truth": "No, this is a logical fallacy (affirming the consequent). We cannot conclude this from the given premises.",
                "difficulty": 3
            },
            {
                "text": "A temperature increase causes pressure to increase in a sealed container. If the pressure decreased, what happened to the temperature?",
                "truth": "The temperature decreased (assuming volume remained constant).",
                "difficulty": 2
            },
            {
                "text": "If it's raining, then the ground is wet. The ground is wet. Is it necessarily raining?",
                "truth": "No, the ground could be wet for other reasons (sprinkler, flood, someone washing, etc.).",
                "difficulty": 3
            },
            {
                "text": "All mammals are warm-blooded. A whale is warm-blooded. Is a whale necessarily a mammal?",
                "truth": "No, being warm-blooded doesn't necessarily make something a mammal. However, whales ARE mammals, but not because of this syllogism.",
                "difficulty": 3
            },
            {
                "text": "If you study hard, you will pass the exam. You passed the exam. Did you study hard?",
                "truth": "Not necessarily. You could have passed for other reasons (prior knowledge, lucky guesses, etc.).",
                "difficulty": 3
            }
        ]
        
        for i, template in enumerate(templates[:count]):
            questions.append(Question(
                id=f"reasoning_{i+1:03d}",
                category=QuestionCategory.REASONING.value,
                domain=Domain.GENERAL.value,
                question_text=template["text"],
                ground_truth=template["truth"],
                difficulty=template["difficulty"],
                requires_calculation=False,
                requires_external_knowledge=False,
                notes="Sample reasoning question"
            ))
        
        return questions
    
    @staticmethod
    def generate_multistep_questions(count: int = 25) -> List[Question]:
        """Generate sample multi-step questions"""
        questions = []
        
        templates = [
            {
                "text": "If you invest $1000 at 5% annual interest compounded annually, how much will you have after 3 years?",
                "truth": "$1157.63 (using formula: A = P(1 + r)^t)",
                "difficulty": 3
            },
            {
                "text": "A train travels 60 mph for 2 hours, then 80 mph for 1.5 hours. What is the average speed?",
                "truth": "68.57 mph (total distance: 240 miles, total time: 3.5 hours)",
                "difficulty": 3
            },
            {
                "text": "A recipe calls for 2 cups of flour for 12 cookies. How much flour is needed for 30 cookies?",
                "truth": "5 cups (proportion: 2/12 = x/30, x = 5)",
                "difficulty": 2
            },
            {
                "text": "If a shirt costs $40 and is on sale for 25% off, then an additional 10% off the sale price, what is the final price?",
                "truth": "$27 (first discount: $30, second discount: $27)",
                "difficulty": 3
            },
            {
                "text": "A car uses 1 gallon of gas to travel 30 miles. How many gallons are needed for a 450-mile trip?",
                "truth": "15 gallons (450 ÷ 30 = 15)",
                "difficulty": 2
            }
        ]
        
        for i, template in enumerate(templates[:count]):
            questions.append(Question(
                id=f"multistep_{i+1:03d}",
                category=QuestionCategory.MULTISTEP.value,
                domain=Domain.GENERAL.value,
                question_text=template["text"],
                ground_truth=template["truth"],
                difficulty=template["difficulty"],
                requires_calculation=True,
                requires_external_knowledge=False,
                notes="Sample multi-step question"
            ))
        
        return questions


# ============================================================================
# ANNOTATION TOOL
# ============================================================================

class AnnotationTool:
    """Interactive tool for annotating model responses"""
    
    def __init__(self, benchmark_path: str):
        """Initialize with a benchmark dataset"""
        with open(benchmark_path, 'r') as f:
            self.data = json.load(f)
        
        self.questions = {q['id']: q for q in self.data['questions']}
        self.responses = self.data.get('responses', [])
        self.annotations = self.data.get('annotations', [])
    
    def annotate_response(self, response_id: str, annotator_id: str) -> Dict:
        """Interactive annotation of a single response"""
        response = next((r for r in self.responses if r['response_id'] == response_id), None)
        if not response:
            return {"error": "Response not found"}
        
        question = self.questions.get(response['question_id'])
        
        print("\n" + "="*70)
        print(f"QUESTION ({question['category']} - {question['domain']}):")
        print(question['question_text'])
        print(f"\nGROUND TRUTH: {question['ground_truth']}")
        print("\n" + "-"*70)
        print(f"MODEL RESPONSE ({response['strategy']}):")
        print(response['response_text'])
        print("="*70)
        
        annotation = {
            'response_id': response_id,
            'question_id': response['question_id'],
            'strategy': response['strategy'],
            'annotator_id': annotator_id
        }
        
        while True:
            has_hallucination = input("\nDoes this response contain hallucinations? (y/n): ").lower()
            if has_hallucination in ['y', 'n']:
                annotation['has_hallucination'] = (has_hallucination == 'y')
                break
            print("Please enter 'y' or 'n'")
        
        while True:
            try:
                correctness = int(input("\nFactual correctness (0-5): "))
                if 0 <= correctness <= 5:
                    annotation['factual_correctness'] = correctness
                    break
                print("Please enter a number between 0 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            try:
                completeness = int(input("\nCompleteness (0-5): "))
                if 0 <= completeness <= 5:
                    annotation['completeness'] = completeness
                    break
                print("Please enter a number between 0 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        if annotation['has_hallucination']:
            details = input("\nDescribe the hallucination(s): ")
            annotation['hallucination_details'] = details
        else:
            annotation['hallucination_details'] = "None"
        
        print("\n✓ Annotation complete!")
        return annotation
    
    def batch_annotate(self, annotator_id: str, strategy: str = None) -> List[Dict]:
        """Annotate multiple responses"""
        annotations = []
        
        responses_to_annotate = self.responses
        if strategy:
            responses_to_annotate = [r for r in self.responses if r['strategy'] == strategy]
        
        print(f"\nStarting batch annotation for {len(responses_to_annotate)} responses")
        
        for i, response in enumerate(responses_to_annotate):
            print(f"\n[{i+1}/{len(responses_to_annotate)}]")
            annotation = self.annotate_response(response['response_id'], annotator_id)
            annotations.append(annotation)
            
            if i < len(responses_to_annotate) - 1:
                cont = input("\nContinue to next response? (y/n): ").lower()
                if cont != 'y':
                    break
        
        return annotations
    
    def save_annotations(self, annotations: List[Dict], filepath: str) -> None:
        """Save annotations to file"""
        self.data['annotations'].extend(annotations)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        print(f"\n✓ Saved {len(annotations)} annotations to {filepath}")


# ============================================================================
# INTER-ANNOTATOR AGREEMENT
# ============================================================================

class InterAnnotatorAgreement:
    """Calculate inter-annotator agreement metrics"""
    
    @staticmethod
    def cohens_kappa(annotations1: List[bool], annotations2: List[bool]) -> float:
        """Calculate Cohen's Kappa for binary classifications"""
        if len(annotations1) != len(annotations2):
            raise ValueError("Annotation lists must be the same length")
        
        n = len(annotations1)
        agreements = sum(a1 == a2 for a1, a2 in zip(annotations1, annotations2))
        po = agreements / n
        
        yes1 = sum(annotations1)
        yes2 = sum(annotations2)
        no1 = n - yes1
        no2 = n - yes2
        
        pe = ((yes1 * yes2) + (no1 * no2)) / (n * n)
        
        if pe == 1:
            return 1.0
        
        kappa = (po - pe) / (1 - pe)
        return kappa
    
    @staticmethod
    def percent_agreement(annotations1: List, annotations2: List) -> float:
        """Calculate simple percent agreement"""
        if len(annotations1) != len(annotations2):
            raise ValueError("Annotation lists must be the same length")
        
        agreements = sum(a1 == a2 for a1, a2 in zip(annotations1, annotations2))
        return agreements / len(annotations1) * 100
    
    @staticmethod
    def analyze_agreement(benchmark_path: str) -> Dict:
        """Analyze inter-annotator agreement for a benchmark"""
        with open(benchmark_path, 'r') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', [])
        
        by_response = defaultdict(list)
        for ann in annotations:
            by_response[ann['response_id']].append(ann)
        
        multi_annotated = {k: v for k, v in by_response.items() if len(v) >= 2}
        
        if not multi_annotated:
            return {"error": "No responses with multiple annotations found"}
        
        results = {
            "total_responses": len(by_response),
            "multi_annotated_responses": len(multi_annotated),
            "hallucination_agreement": {},
            "correctness_agreement": {},
            "completeness_agreement": {}
        }
        
        hall_ann1, hall_ann2 = [], []
        corr_ann1, corr_ann2 = [], []
        comp_ann1, comp_ann2 = [], []
        
        for response_id, anns in multi_annotated.items():
            ann1, ann2 = anns[0], anns[1]
            
            hall_ann1.append(ann1['has_hallucination'])
            hall_ann2.append(ann2['has_hallucination'])
            
            corr_ann1.append(ann1['factual_correctness'])
            corr_ann2.append(ann2['factual_correctness'])
            
            comp_ann1.append(ann1['completeness'])
            comp_ann2.append(ann2['completeness'])
        
        kappa = InterAnnotatorAgreement.cohens_kappa(hall_ann1, hall_ann2)
        percent = InterAnnotatorAgreement.percent_agreement(hall_ann1, hall_ann2)
        
        results["hallucination_agreement"] = {
            "cohens_kappa": round(kappa, 3),
            "percent_agreement": round(percent, 2),
            "interpretation": InterAnnotatorAgreement._interpret_kappa(kappa)
        }
        
        exact_match = InterAnnotatorAgreement.percent_agreement(corr_ann1, corr_ann2)
        within_1 = sum(abs(c1 - c2) <= 1 for c1, c2 in zip(corr_ann1, corr_ann2)) / len(corr_ann1) * 100
        
        results["correctness_agreement"] = {
            "exact_match": round(exact_match, 2),
            "within_1_point": round(within_1, 2),
            "mean_difference": round(statistics.mean([abs(c1 - c2) for c1, c2 in zip(corr_ann1, corr_ann2)]), 2)
        }
        
        exact_match = InterAnnotatorAgreement.percent_agreement(comp_ann1, comp_ann2)
        within_1 = sum(abs(c1 - c2) <= 1 for c1, c2 in zip(comp_ann1, comp_ann2)) / len(comp_ann1) * 100
        
        results["completeness_agreement"] = {
            "exact_match": round(exact_match, 2),
            "within_1_point": round(within_1, 2),
            "mean_difference": round(statistics.mean([abs(c1 - c2) for c1, c2 in zip(comp_ann1, comp_ann2)]), 2)
        }
        
        return results
    
    @staticmethod
    def _interpret_kappa(kappa: float) -> str:
        """Interpret Cohen's Kappa value"""
        if kappa < 0:
            return "Poor (less than chance agreement)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost perfect"


# ============================================================================
# ANNOTATION ANALYZER
# ============================================================================

class AnnotationAnalyzer:
    """Analyze annotation results and generate statistics"""
    
    @staticmethod
    def calculate_hallucination_rates(benchmark_path: str) -> Dict:
        """Calculate hallucination rates by strategy and category"""
        with open(benchmark_path, 'r') as f:
            data = json.load(f)
        
        questions = {q['id']: q for q in data['questions']}
        annotations = data.get('annotations', [])
        
        by_strategy = defaultdict(list)
        for ann in annotations:
            by_strategy[ann['strategy']].append(ann)
        
        results = {}
        
        for strategy, anns in by_strategy.items():
            hallucination_count = sum(a['has_hallucination'] for a in anns)
            total = len(anns)
            rate = (hallucination_count / total * 100) if total > 0 else 0
            
            by_category = defaultdict(lambda: {'hall': 0, 'total': 0})
            for ann in anns:
                question = questions[ann['question_id']]
                category = question['category']
                by_category[category]['total'] += 1
                if ann['has_hallucination']:
                    by_category[category]['hall'] += 1
            
            category_rates = {}
            for cat, counts in by_category.items():
                category_rates[cat] = round(counts['hall'] / counts['total'] * 100, 2)
            
            results[strategy] = {
                'overall_rate': round(rate, 2),
                'total_responses': total,
                'hallucinations': hallucination_count,
                'by_category': category_rates
            }
        
        return results
    
    @staticmethod
    def calculate_average_scores(benchmark_path: str) -> Dict:
        """Calculate average correctness and completeness scores"""
        with open(benchmark_path, 'r') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', [])
        
        by_strategy = defaultdict(lambda: {'correctness': [], 'completeness': []})
        
        for ann in annotations:
            strategy = ann['strategy']
            by_strategy[strategy]['correctness'].append(ann['factual_correctness'])
            by_strategy[strategy]['completeness'].append(ann['completeness'])
        
        results = {}
        for strategy, scores in by_strategy.items():
            results[strategy] = {
                'avg_correctness': round(statistics.mean(scores['correctness']), 2),
                'avg_completeness': round(statistics.mean(scores['completeness']), 2),
                'std_correctness': round(statistics.stdev(scores['correctness']), 2) if len(scores['correctness']) > 1 else 0,
                'std_completeness': round(statistics.stdev(scores['completeness']), 2) if len(scores['completeness']) > 1 else 0
            }
        
        return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_benchmark_command(args):
    """Create a sample benchmark"""
    print("\n" + "="*70)
    print("Creating Sample Benchmark")
    print("="*70 + "\n")
    
    benchmark = BenchmarkDataset("hallucination_benchmark")
    
    generator = QuestionGenerator()
    benchmark.questions.extend(generator.generate_factual_questions(10))
    benchmark.questions.extend(generator.generate_reasoning_questions(5))
    benchmark.questions.extend(generator.generate_multistep_questions(5))
    
    print(f"✓ Created benchmark with {len(benchmark.questions)} questions")
    
    stats = benchmark.get_statistics()
    print("\nBenchmark Statistics:")
    print(f"  Total questions: {stats['total_questions']}")
    print(f"  By category: {stats['questions_by_category']}")
    print(f"  By domain: {stats['questions_by_domain']}")
    
    benchmark.save_to_json("benchmark_dataset.json")
    benchmark.export_questions_csv("questions.csv")
    
    print("\n✓ Files created:")
    print("  - benchmark_dataset.json (main dataset)")
    print("  - questions.csv (for easy editing)")


def annotate_command(args):
    """Start annotation session"""
    print("\n" + "="*70)
    print("Annotation Session")
    print("="*70 + "\n")
    
    annotator_id = input("Enter your annotator ID: ")
    
    tool = AnnotationTool("benchmark_dataset.json")
    annotations = tool.batch_annotate(annotator_id)
    tool.save_annotations(annotations, "benchmark_dataset.json")


def analyze_command(args):
    """Analyze results"""
    print("\n" + "="*70)
    print("Analysis Results")
    print("="*70 + "\n")
    
    print("Hallucination Rates by Strategy:")
    print("-" * 70)
    rates = AnnotationAnalyzer.calculate_hallucination_rates("benchmark_dataset.json")
    print(json.dumps(rates, indent=2))
    
    print("\n\nAverage Scores:")
    print("-" * 70)
    scores = AnnotationAnalyzer.calculate_average_scores("benchmark_dataset.json")
    print(json.dumps(scores, indent=2))


def agreement_command(args):
    """Calculate inter-annotator agreement"""
    print("\n" + "="*70)
    print("Inter-Annotator Agreement")
    print("="*70 + "\n")
    
    agreement = InterAnnotatorAgreement.analyze_agreement("benchmark_dataset.json")
    print(json.dumps(agreement, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Hallucination Benchmark Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hallucination_benchmark.py --create      # Create sample benchmark
  python hallucination_benchmark.py --annotate    # Start annotation
  python hallucination_benchmark.py --analyze     # Analyze results
  python hallucination_benchmark.py --agreement   # Check agreement
        """
    )
    
    parser.add_argument('--create', action='store_true', help='Create sample benchmark')
    parser.add_argument('--annotate', action='store_true', help='Start annotation session')
    parser.add_argument('--analyze', action='store_true', help='Analyze results')
    parser.add_argument('--agreement', action='store_true', help='Calculate inter-annotator agreement')
    
    args = parser.parse_args()
    
    if args.create:
        create_benchmark_command(args)
    elif args.annotate:
        annotate_command(args)
    elif args.analyze:
        analyze_command(args)
    elif args.agreement:
        agreement_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
