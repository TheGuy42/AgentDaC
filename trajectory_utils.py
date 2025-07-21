from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from art import Trajectory


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class TrajectoryNode:
    trajectory: Trajectory
    parent: Optional["TrajectoryNode"] = None
    children: Optional[list["TrajectoryNode"]] = field(default_factory=list)

    def print(self, indent: int = 0):
        """
        Print the trajectory node and its children with indentation.
        """
        repr = self._repr(indent=indent)
        print(repr)

    def _repr(self, indent: int = 0) -> str:
        """
        Return a string representation of the trajectory node.
        """
        sep = "" if indent == 0 else "|  "
        repr = ""
        for message in self.trajectory.messages():
            role = f"{bcolors.OKGREEN}{message['role']}{bcolors.ENDC}:: "
            content = message["content"]
            content = content.replace("<task>", f"{bcolors.OKCYAN}<task>{bcolors.ENDC}")
            content = content.replace("</task>", f"{bcolors.OKCYAN}</task>{bcolors.ENDC}")
            content = content.replace("<answer>", f"{bcolors.OKBLUE}<answer>{bcolors.ENDC}")
            content = content.replace("</answer>", f"{bcolors.OKBLUE}</answer>{bcolors.ENDC}")

            # print(f"{role}{content}")
            repr += f"{role}{content}\n"

        repr = "\n".join([f"{'  ' * (indent - 1)}{sep}{line}" for line in repr.splitlines()])
        return repr
