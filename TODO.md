# TODOs

0. In the nothink prompts, we should replace private reasoning to internal reasoning. This is more standard.

1. If we use a reasoning-parser then the regex / GuidedDecoding will be applied only to the post-reasoning output. This allows the model to first emit `<think>` `<\think>` block and only then the GuidedDecoding regex will be applied. Otherwise, GuidedDecoding is applied immediately after the model emits the first token.

2. The above allows us to simply remove the think action completly, and simply allow the model to perform the remaining actions. This way the model will always have the opportunity to first reason, and only then to perform some action. Maybe we should add an action called continue which will allow the model to continue reasoning.
