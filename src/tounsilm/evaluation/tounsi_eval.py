"""
TounsiEval: Benchmark Suite for Tunisian Language Models.

Contains 14 specialized categories:
  1. Tunisian vocabulary
  2. Tunisian grammar
  3. Tunisian comprehension
  4. Tunisian generation
  5. Arabizi comprehension
  6. Arabizi generation
  7. Arabic-script Tunisian
  8. Tunisian/French code-switching
  9. Translation
  10. Everyday conversation
  11. Technical Tunisian
  12. Reasoning in Tunisian
  13. Cultural/contextual understanding
  14. Robustness to spelling variation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalSample:
    id: str
    category: str
    input: str
    expected_behavior: str
    reference_answer: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalSample":
        return cls(
            id=data["id"],
            category=data["category"],
            input=data["input"],
            expected_behavior=data.get("expected_behavior", ""),
            reference_answer=data.get("reference_answer", ""),
        )


TOUNSI_EVAL_BENCHMARK: List[EvalSample] = [
    # 1. Tunisian vocabulary
    EvalSample(
        id="vocab-01",
        category="Tunisian vocabulary",
        input="شنوة معنى كلمة 'برشا' في الدارجة التونسية؟",
        expected_behavior="Explain meaning of 'barsha' (a lot / very much)",
        reference_answer="معناها بزاف، كثير، أو بكمية كبيرة.",
    ),
    EvalSample(
        id="vocab-02",
        category="Tunisian vocabulary",
        input="شنوة الفرق بين 'فيسع' و'بالشويا'؟",
        expected_behavior="Distinguish fast vs slow adverbs",
        reference_answer="'فيسع' معناها بسرعة، و'بالشويا' معناها ببطء أو على مهل.",
    ),
    # 2. Tunisian grammar
    EvalSample(
        id="grammar-01",
        category="Tunisian grammar",
        input="صرف فعل 'مشى' مع الضمير 'أحنا' في الماضي بالدارجة التونسية.",
        expected_behavior="Conjugate 'msha' for 'we' in past tense",
        reference_answer="أحنا مشينا.",
    ),
    EvalSample(
        id="grammar-02",
        category="Tunisian grammar",
        input="كيفاش ننفي جملة 'هو يكتب' بالدارجة التونسية؟",
        expected_behavior="Apply ma-...-sh negation pattern",
        reference_answer="هو ما يكتبش.",
    ),
    # 3. Tunisian comprehension
    EvalSample(
        id="comp-01",
        category="Tunisian comprehension",
        input="كي واحد يقلك 'يعيشك ويرحم والديك'، شنوة يقصد؟",
        expected_behavior="Understand common gratitude expression",
        reference_answer="يقصد شكر ودعاء بالخير والرحمة للوالدين.",
    ),
    # 4. Tunisian generation
    EvalSample(
        id="gen-01",
        category="Tunisian generation",
        input="اكتب جملة تونسية ترحّب بيها بضيف جاك للدار.",
        expected_behavior="Produce natural Tunisian welcome",
        reference_answer="مرحبا بيك، نورتنا والدار دارك تفضل.",
    ),
    # 5. Arabizi comprehension
    EvalSample(
        id="arabizi-comp-01",
        category="Arabizi comprehension",
        input="chnowa ma3naha: '3lech ma jitich lbare7 lel 5edma?'",
        expected_behavior="Comprehend question in Arabizi",
        reference_answer="علاش ما جيتش البارح للخدمة؟ السؤال يستفسر على سبب الغياب.",
    ),
    # 6. Arabizi generation
    EvalSample(
        id="arabizi-gen-01",
        category="Arabizi generation",
        input="a3tini jomla bel arabizi t9oul fiha lsa7bek bech yet9ablou fel 9ahwa.",
        expected_behavior="Generate conversational Arabizi café invitation",
        reference_answer="winek ya 5ouya, ch9awlek net9ablou fel 9ahwa lila?",
    ),
    # 7. Arabic-script Tunisian
    EvalSample(
        id="script-01",
        category="Arabic-script Tunisian",
        input="توّا وقتاش باش تبداو الخدمة على المشروع الجديد؟",
        expected_behavior="Natural response in Tunisian Arabic script",
        reference_answer="باش نبداو نخدمو عليه الجمعة الجاية انشالله.",
    ),
    # 8. Tunisian/French code-switching
    EvalSample(
        id="codeswitch-01",
        category="Tunisian/French code-switching",
        input="كملت الـréunion متاعك ولا مزال عندك des tâches أخرى؟",
        expected_behavior="Process and answer code-switched question naturally",
        reference_answer="كملت الـréunion وتوّا باش نبدا في الـtâches الأخرين.",
    ),
    # 9. Translation
    EvalSample(
        id="trans-01",
        category="Translation",
        input="ترجم للعربية التونسية: 'Where are you going right now?'",
        expected_behavior="Translate English into Derja",
        reference_answer="وين ماشي توّا؟ / وينك ماشي توّا؟",
    ),
    EvalSample(
        id="trans-02",
        category="Translation",
        input="ترجم من التونسي للفرنسية: 'ما نجمش نجي اليوم خاطر مريض.'",
        expected_behavior="Translate Derja into French",
        reference_answer="Je ne peux pas venir aujourd'hui parce que je suis malade.",
    ),
    # 10. Everyday conversation
    EvalSample(
        id="conv-01",
        category="Everyday conversation",
        input="سلام، شحوالك ولاباس عليك؟",
        expected_behavior="Natural polite conversational greeting reply",
        reference_answer="وعليكم السلام، الحمد لله لاباس، يعيشك وأنت شنوة أحوالك؟",
    ),
    # 11. Technical Tunisian
    EvalSample(
        id="tech-01",
        category="Technical Tunisian",
        input="اشرحلي بالتونسي شنوة هو الـAPI وكيفاش يخدم.",
        expected_behavior="Explain API concept clearly in Tunisian dialect",
        reference_answer="الـAPI هو كيف الوسيط بين زوز برامج يخليهم يتواصلو ويبعثو données لبعضهم.",
    ),
    # 12. Reasoning in Tunisian
    EvalSample(
        id="reason-01",
        category="Reasoning in Tunisian",
        input="سامي عندو 5 كعبات ملاوي، عطى لمراد زوز وكلا كعبة. قداش قعدلو من كعبة؟",
        expected_behavior="Solve word problem in Tunisian (5 - 2 - 1 = 2)",
        reference_answer="قعدلو زوز (2) كعبات ملاوي.",
    ),
    # 13. Cultural/contextual understanding
    EvalSample(
        id="culture-01",
        category="Cultural/contextual understanding",
        input="شنوة هي الأكلة التونسية التقليدية المشهورة يوم الجمعة؟",
        expected_behavior="Identify Tunisian Friday tradition (Couscous)",
        reference_answer="الكسكسي هو الأكلة التقليدية المعروفة برشا نهار الجمعة في تونس.",
    ),
    # 14. Robustness to spelling variation
    EvalSample(
        id="robust-01",
        category="Robustness to spelling variation",
        input="شنوا الفرق في المعنى بين 'شنوة تعمل' و 'شعملت' و 'chneya ta3mel'؟",
        expected_behavior="Recognize temporal and orthographic variations of same root concepts",
        reference_answer="'شنوة تعمل' و 'chneya ta3mel' في الحاضر، و'شعملت' في الماضي.",
    ),
]


def save_benchmark_file(filepath: str | Path) -> None:
    """Save the standard TounsiEval benchmark to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in TOUNSI_EVAL_BENCHMARK:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def load_benchmark_file(filepath: str | Path) -> List[EvalSample]:
    """Load benchmark dataset from JSONL."""
    path = Path(filepath)
    if not path.exists():
        save_benchmark_file(path)
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(EvalSample.from_dict(json.loads(line.strip())))
    return samples
