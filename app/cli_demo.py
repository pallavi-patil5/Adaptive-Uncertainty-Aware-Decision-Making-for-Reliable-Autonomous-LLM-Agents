# app/cli_demo.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from policy.agent_graph import build_graph

def main():
    graph = build_graph()
    print("Adaptive Uncertainty-Aware Agent — CLI Demo")
    print("Type a question, or 'quit' to exit.\n")

    while True:
        question = input("Q: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        state = graph.invoke({
            "question": question, "features": None,
            "decision": None, "result": None, "evidence": None,
        })

        features = state["features"]
        decision = state["decision"]
        result = state["result"]

        print(f"\n  Uncertainty: {features['uncertainty']:.3f} | "
              f"Ambiguity: {features['ambiguity']:.3f} | "
              f"Contradiction: {features['contradiction_prob']:.3f} | "
              f"Evidence Coverage: {features['evidence_coverage']:.3f}")
        print(f"  Action scores: { {k: round(v,3) for k,v in decision['scores'].items()} }")
        print(f"  -> ACTION: {decision['action'].upper()}")
        print(f"  Answer: {result.get('final_answer', '(none)')}\n")


if __name__ == "__main__":
    main()