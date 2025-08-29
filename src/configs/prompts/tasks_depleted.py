from src.utils.markers import Markers as M
from src.configs.prompts import add_prompt


add_prompt(
    name="tasks_depleted",
    content=f"""
Task budget depleted - no more tasks available, you can't create more tasks via {M.TASK_START}. 
Instead, you must provide the final answer in an {M.ANS_START} block.
""",
)

add_prompt(
    name="tasks_depleted_v2",
    content=f"""
WARNING: Sub-task budget depleted - no more tasks available. You can't create sub-tasks anymore. 
Instead, you must think by yourself and then provide the final answer in an {M.ANS_START} {M.ANS_END} block.
""",
)
