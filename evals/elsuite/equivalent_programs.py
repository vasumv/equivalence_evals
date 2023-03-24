import evals
import evals.metrics

task_description = """
Determine if the following two programs are semantically equivalent. Response is "True"/"False". No explanation.
"""
programs = """
```
{first}
```
```
{second}
```
"""

class EquivalentProgram(evals.Eval):

    def __init__(self, samples_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.samples_jsonl = samples_jsonl

    def run(self, recorder):
        """
        Called by the `oaieval` CLI to run the eval. The `eval_all_samples` method calls `eval_sample`.
        """
        test_samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, test_samples)

        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }

    def eval_sample(self, sample, rng):
        prompt = [
            {"role": "system", "content": task_description}
        ]
        first_program, second_program = sample["first"], sample["second"]
        prompt += [
            {"role": "user", "content": f"```{first_program}```\n```{second_program}```"}
        ]
        evals.check_sampled_text(self.model_spec, prompt, expected=sample["equivalence"])

