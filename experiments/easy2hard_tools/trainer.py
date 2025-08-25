from src.dac_agent_tools import AgentToolNode
from src.dac_agent import AgentNode
from src.trainer import RolloutStage

from experiments.easy2hard_ruler.trainer import Easy2HardRulerTrainer


class Easy2HardToolTrainer(Easy2HardRulerTrainer):
    def create_agent(self, stage: RolloutStage) -> AgentNode:
        client = self.vllm_router.next()
        return AgentToolNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
        )
