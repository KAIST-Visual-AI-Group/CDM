"""Optional generation-quality metric: perplexity of the generations under Qwen2.5.

Only used when `eval_ppl=true`; the `evaluate` package is imported lazily so the rest of the
app does not depend on it.
"""

class QwenPPL():
    def __init__(self, model_name=None, task="text"):
        self.task = task

        if model_name is not None:
            self.model_name = model_name
        # if model name not specified use based on task (Qwen2.5 variants, 3B size)
        else:
            if self.task == "text":
                self.model_name = "Qwen/Qwen2.5-3B-Instruct"
            elif self.task == "code":
                self.model_name = "Qwen/Qwen3-4B-Instruct-2507"

        from evaluate import load

        self.perplexity = load("perplexity", module_type="metric")

    # text is a list of decoded strings
    # returns a list of perplexity values for each string
    def __call__(self, text):
        try:
            result = self.perplexity.compute(
                model_id=self.model_name,
                add_start_token=False,
                predictions=text,
            )
            return result["perplexities"]
        except AssertionError:
            # Fallback: skip invalid short strings instead of crashing the run.
            ppls = []
            for t in text:
                try:
                    one = self.perplexity.compute(
                        model_id=self.model_name,
                        add_start_token=False,
                        predictions=[t],
                    )
                    ppls.append(one["perplexities"][0])
                except AssertionError:
                    ppls.append(float("nan"))
            return ppls
