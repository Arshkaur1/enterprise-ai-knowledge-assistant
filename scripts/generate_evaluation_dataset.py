import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data/evaluation/retrieval_eval.jsonl"

HR0 = "hr_policy_chunk_0000"
HR1 = "hr_policy_chunk_0001"

IT0 = "it_security_chunk_0000"
IT1 = "it_security_chunk_0001"

FIN0 = "finance_policy_p1_chunk_0000"
FIN1 = "finance_policy_p1_chunk_0001"


def record(
    query_id,
    query,
    category,
    relevant_chunks,
    expected_answer,
    split,
):
    return {
        "query_id": query_id,
        "query": query,
        "category": category,
        "answerable": bool(relevant_chunks),
        "relevant_chunk_ids": relevant_chunks,
        "expected_answer": expected_answer,
        "split": split,
    }


def generate_dataset():
    return [
        # HR policy: 6 questions
        record(
            "hr_001",
            "How many days per week can employees work remotely?",
            "HR",
            [HR0],
            "Up to three days per week.",
            "calibration",
        ),
        record(
            "hr_002",
            "How much annual paid leave do full-time employees receive?",
            "HR",
            [HR0],
            "20 working days per calendar year.",
            "calibration",
        ),
        record(
            "hr_003",
            "How far in advance should annual leave be requested?",
            "HR",
            [HR0],
            "At least five working days in advance.",
            "calibration",
        ),
        record(
            "hr_004",
            "When should an employee notify their manager about illness?",
            "HR",
            [HR1],
            "As soon as reasonably possible.",
            "calibration",
        ),
        record(
            "hr_005",
            "What benefits are available to eligible full-time employees?",
            "HR",
            [HR1],
            "Health coverage and professional development programs.",
            "test",
        ),
        record(
            "hr_006",
            "Can employees take leave when illness prevents them from working?",
            "HR",
            [HR0, HR1],
            "Yes, employees may take sick leave.",
            "test",
        ),
        # IT security: 6 questions
        record(
            "it_001",
            "What is the minimum password length?",
            "IT",
            [IT0],
            "At least 12 characters.",
            "calibration",
        ),
        record(
            "it_002",
            "Is multi-factor authentication mandatory?",
            "IT",
            [IT0],
            "Yes, multi-factor authentication is mandatory.",
            "calibration",
        ),
        record(
            "it_003",
            "Who must approve employee access requests?",
            "IT",
            [IT0],
            "Managers must approve access requests.",
            "calibration",
        ),
        record(
            "it_004",
            "What information should a security incident report include?",
            "IT",
            [IT0, IT1],
            "Approximate time, affected systems and relevant observations.",
            "calibration",
        ),
        record(
            "it_005",
            "Can company data be stored in unauthorized personal storage?",
            "IT",
            [IT1],
            "No, such transfers are prohibited.",
            "test",
        ),
        record(
            "it_006",
            "What should employees do to keep unattended devices secure?",
            "IT",
            [IT1],
            "Lock their devices when unattended.",
            "test",
        ),
        # Finance policy: 6 questions
        record(
            "finance_001",
            "What is the daily meal reimbursement limit?",
            "Finance",
            [FIN0],
            "40 currency units per employee per day.",
            "calibration",
        ),
        record(
            "finance_002",
            "What is the maximum accommodation allowance per night?",
            "Finance",
            [FIN0],
            "150 currency units per night.",
            "calibration",
        ),
        record(
            "finance_003",
            "What is the deadline for submitting expense claims?",
            "Finance",
            [FIN0],
            "Within 30 calendar days of the transaction.",
            "calibration",
        ),
        record(
            "finance_004",
            "Who approves claims below 500 currency units?",
            "Finance",
            [FIN0, FIN1],
            "The employee's manager.",
            "calibration",
        ),
        record(
            "finance_005",
            "Who must approve an expense claim of 500 currency units?",
            "Finance",
            [FIN1],
            "Both the manager and Finance department.",
            "test",
        ),
        record(
            "finance_006",
            "Are personal expenses eligible for reimbursement?",
            "Finance",
            [FIN0],
            "No, personal expenses are not eligible.",
            "test",
        ),
        # Multi-chunk questions: 4 questions
        record(
            "multi_001",
            "How much annual leave is provided and what medical "
            "documentation may be required for extended sick leave?",
            "multi_chunk",
            [HR0, HR1],
            "20 working days of annual leave. Medical documentation "
            "may be requested for extended absences.",
            "calibration",
        ),
        record(
            "multi_002",
            "What is the expense submission deadline and who approves "
            "claims of 500 currency units or more?",
            "multi_chunk",
            [FIN0, FIN1],
            "Within 30 calendar days. Both the manager and Finance "
            "department must approve claims of 500 or more.",
            "calibration",
        ),
        record(
            "multi_003",
            "What is the minimum password length and how should "
            "unattended devices be secured?",
            "multi_chunk",
            [IT0, IT1],
            "Passwords must contain at least 12 characters. "
            "Unattended devices must be locked.",
            "test",
        ),
        record(
            "multi_004",
            "How many days can employees work remotely and what "
            "is the daily meal reimbursement limit?",
            "multi_chunk",
            [HR0, FIN0],
            "Up to three remote working days per week and "
            "40 currency units per employee per day for meals.",
            "test",
        ),
        # Completely unrelated questions: 4 questions
        record(
            "ood_001",
            "What is the weather forecast for tomorrow?",
            "out_of_domain",
            [],
            None,
            "calibration",
        ),
        record(
            "ood_002",
            "Who invented the telephone?",
            "out_of_domain",
            [],
            None,
            "calibration",
        ),
        record(
            "ood_003",
            "How do I bake a chocolate cake?",
            "out_of_domain",
            [],
            None,
            "test",
        ),
        record(
            "ood_004",
            "What is the capital of Japan?",
            "out_of_domain",
            [],
            None,
            "test",
        ),
        # Domain-related but unanswerable: 4 questions
        record(
            "unsupported_001",
            "How many paid maternity leave days are provided?",
            "unsupported_policy",
            [],
            None,
            "calibration",
        ),
        record(
            "unsupported_002",
            "How frequently must employees change their passwords?",
            "unsupported_policy",
            [],
            None,
            "calibration",
        ),
        record(
            "unsupported_003",
            "What is the company's mileage reimbursement rate?",
            "unsupported_policy",
            [],
            None,
            "test",
        ),
        record(
            "unsupported_004",
            "How much annual salary increase do employees receive?",
            "unsupported_policy",
            [],
            None,
            "test",
        ),
    ]


def main():
    records = generate_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} evaluation questions.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
