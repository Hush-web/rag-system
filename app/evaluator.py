import time
from rag import RAGSystem
from test_set import test_questions
from loguru import logger

class Evaluator:
    def __init__(self):
        self.rag = RAGSystem()

    def check_answer(self, answer, expected, keywords):
        answer_lower = answer.lower()
        matches = [kw for kw in keywords if kw.lower() in answer_lower]
        score = len(matches) / len(keywords) if keywords else 0
        return score, matches

    def run_tests(self):
        results = []
        total_score = 0

        for i, test in enumerate(test_questions):
            print(f"\n{'='*60}")
            print(f"Test {i+1}: {test['question']}")
            print(f"{'='*60}")

            result = self.rag.answer(test['question'])
            answer = result['answer']

            score, matches = self.check_answer(
                answer,
                test['expected'],
                test['keywords']
            )

            results.append({
                "question": test['question'],
                "answer": answer[:200] + "...",
                "expected_keywords": test['keywords'],
                "matched_keywords": matches,
                "score": score
            })
            total_score += score

            print(f"Answer: {answer[:300]}...")
            print(f"Matched keywords: {matches}")
            print(f"Score: {score:.0%}")

            # Wait 2 seconds to avoid rate limiting
            time.sleep(2)

        avg_score = total_score / len(test_questions)

        print(f"\n{'='*60}")
        print(f"OVERALL ACCURACY: {avg_score:.0%}")
        print(f"{'='*60}")

        return results, avg_score

if __name__ == "__main__":
    evaluator = Evaluator()
    results, accuracy = evaluator.run_tests()